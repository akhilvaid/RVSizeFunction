# Neural network to regress RVEF values

import os

import tqdm
import torch
import torchvision

import pandas as pd

from PIL import Image
from sklearn import model_selection, metrics

from config import Config

tqdm.tqdm.pandas()


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

        # Use imagenet mean/median by default
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(
                Config.image_size,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std)])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        # For labels
        ecgid = self.df.iloc[index]['eid']

        # Expects 2 columns: PATH / OUTCOME
        image_path = self.df.iloc[index]['ecg']
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        label = self.df.iloc[index]['label']
        label = torch.tensor(label, dtype=torch.float32)

        return image, label, ecgid


class UKBBNN:
    def __init__(self, outcome, is_classification):
        self.outcome = outcome
        self.is_classification = is_classification
        self.outdir = None

    def gaping_maw(self, train_dataloader, eval_dataloaders):
        all_results = []

        weights = torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
        model = torchvision.models.densenet201(weights=weights)
        model.classifier = torch.nn.Linear(1920, 1)

        # Move to GPU and Dataparallel
        if not Config.single_gpu:
            model = torch.nn.DataParallel(model)
        model = model.cuda()

        # Define loss function
        criterion = torch.nn.BCEWithLogitsLoss() if self.is_classification else torch.nn.MSELoss()

        # Define optimizer, scaler and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=Config.ft_epochs,
            steps_per_epoch=len(train_dataloader))

        # Train
        for epoch in range(Config.ft_epochs + Config.additional_epochs):
            epoch_loss = 0
            model.train()

            for images, labels, _ in tqdm.tqdm(train_dataloader):

                with torch.cuda.amp.autocast():
                    images = images.cuda()
                    labels = labels.cuda()

                    # Same as optim.zero_grad()
                    for param in model.parameters():
                        param.grad = None

                    # Forward pass
                    outputs = model(images)

                    # Calculate loss
                    loss = criterion(outputs.squeeze(), labels.to(torch.float32))
                    epoch_loss += loss.item() * outputs.shape[0]

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # Continue training for 5 epochs more
                    if epoch < Config.ft_epochs:
                        scheduler.step()

            # Overall epoch loss
            training_epoch_loss = epoch_loss / len(train_dataloader)

            # Save model
            torch.save(model.state_dict(), f'{self.modeldir}/model_{epoch}.pt')

            # Evaluate
            model.eval()
            for desc, eval_dataloader in eval_dataloaders.items():
                all_preds = []
                all_labels = []
                all_ecgids = []

                for images, labels, ecgids in tqdm.tqdm(eval_dataloader):
                    testing_epoch_loss = 0

                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            images = images.cuda()
                            labels = labels.cuda()

                            outputs = model(images)

                            # Calculate loss - probably not required
                            testing_loss = criterion(outputs.squeeze(), labels.to(torch.float32))
                            testing_epoch_loss += testing_loss.item() * outputs.shape[0]

                            # Put all outputs together
                            if self.is_classification:
                                all_preds.extend(torch.sigmoid(outputs.squeeze()).cpu().numpy().tolist())
                            else:
                                all_preds.extend(outputs.squeeze().cpu().numpy())
    
                            all_labels.extend(labels.cpu().numpy())
                            all_ecgids.extend(ecgids)

                df_pred = pd.DataFrame({'ecgid': all_ecgids, 'pred': all_preds, 'label': all_labels})

                if self.is_classification:
                    metric = metrics.roc_auc_score(df_pred['label'], df_pred['pred'])
                else:
                    metric = metrics.mean_absolute_error(df_pred['label'], df_pred['pred'])

                all_results.append([desc, epoch, training_epoch_loss, testing_epoch_loss, metric])
                print(f'Epoch: {epoch}, Training Loss: {training_epoch_loss}, {desc.upper()} Loss: {testing_epoch_loss}, METRIC: {metric}')

                # Save predictions
                outfile_name = f'{self.outdir}/epoch_{epoch}.pickle'
                df_pred.to_pickle(outfile_name)

        df_results = pd.DataFrame(all_results, columns=['eval_mode', 'epoch', 'training_loss', 'testing_loss', 'metric'])
        df_results.to_pickle(f'Results/{self.outcome}_{self.is_classification}_Results.pickle')

    def create_dataloaders(self):
        # Load data
        df = pd.read_pickle(Config.df_final)
        df['ecg'] = df['ecg'].apply(lambda x: os.path.join(Config.dir_ecgs, x))
        df['keep'] = df['ecg'].apply(lambda x: os.path.exists(x))
        df = df[df['keep']]
        df = df.drop(columns=['keep'])

        # Cleanup: RVEF between 10 and 80
        df = df.query('10 < RVEF < 80')
        df = df.reset_index(drop=True)

        if self.outcome == 'RVEF':
            if self.is_classification:
                df['label'] = df['RVEF'].apply(lambda x: 1 if x <= Config.rvef_classification else 0)
            else:
                df['label'] = df['RVEF']

        if self.outcome == 'RVEDV_BSA':
            if self.is_classification:
                df['label'] = df['RVEDV_BSA'].apply(lambda x: 1 if x <= Config.rvedv_classification else 0)
            else:
                df['label'] = df['RVEDV_BSA']

        # Group shuffle split
        gss = model_selection.GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=Config.random_state)
        train_idx, test_idx = next(gss.split(df, groups=df['eid']))

        train = df.iloc[train_idx]
        train, validation = model_selection.train_test_split(
            train, test_size=0.05, random_state=Config.random_state)
        test = df.iloc[test_idx]

        if Config.debug:
            train = train.sample(100)
            validation = validation.sample(100)
            test = test.sample(100)

        print()
        print(self.outcome, 'Classification:', self.is_classification)
        print('Train', train.shape, 'Validation', validation.shape, 'Test', test.shape)

        # Create dataloaders
        train_dataset = ECGDataset(train)
        val_dataset = ECGDataset(validation)
        test_dataset = ECGDataset(test)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=Config.batch_size, shuffle=True,
            num_workers=40, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=Config.batch_size, shuffle=False,
            num_workers=40, pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=Config.batch_size, shuffle=False,
            num_workers=40, pin_memory=True)

        return train_dataloader, val_dataloader, test_dataloader

    def hammer_time(self):
        # Housekeeping
        self.outdir = f'Results/{self.outcome}/Classification' if self.is_classification else f'Results/{self.outcome}/Regression'
        os.makedirs(self.outdir, exist_ok=True)
        
        self.modeldir = f'Models/{self.outcome}/Classification' if self.is_classification else f'Models/{self.outcome}/Regression'
        os.makedirs(self.modeldir, exist_ok=True)

        # Proceed as usual
        train_dataloader, val_dataloader, test_dataloader = self.create_dataloaders()

        eval_dataloaders = {
            'validation': val_dataloader,
            'test': test_dataloader}

        self.gaping_maw(train_dataloader, eval_dataloaders)
