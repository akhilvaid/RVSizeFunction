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
    def __init__(self, df, return_full_path=False):
        self.df = df
        self.return_full_path = return_full_path

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
        # Expects 2 columns: PATH / OUTCOME
        image_path = self.df.iloc[index]['FILENAME']
        return_path = image_path if self.return_full_path else image_path.split('/')[-1].split('.')[0]
        image = Image.open(image_path).convert('RGB')
        image = self.transforms(image)

        label = self.df.iloc[index]['label']
        label = torch.tensor(label, dtype=torch.long)

        return image, label, return_path


class UKBBNN:
    def __init__(self, outcome, is_classification, baseline_model, random_baseline=False):
        self.outcome = outcome

        self.is_classification = is_classification
        self.class_reg_string = 'Classification' if is_classification else 'Regression'
        self.random_baseline = random_baseline
        
        self.baseline_model = baseline_model
        self.outdir = None
        print(f'Outcome: {self.outcome}', self.class_reg_string)
        print('Baseline model:', self.baseline_model)

    def load_existing_model(self, model_path):
        print('Loading existing model:', model_path)
        state_dict = torch.load(model_path)

        try:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        except KeyError:
            pass  # In case the model was saved without DataParallel

        return state_dict
    
    def create_model(self, output_neurons):
        # Create model
        if self.baseline_model is None:
            model = torchvision.models.densenet201(weights=None)
            if not self.random_baseline:  # No need to load weights if random baseline
                state_dict = self.load_existing_model(Config.file_lvef40_densenet201)
                model.classifier = torch.nn.Linear(1920, 1)
                model.load_state_dict(state_dict, strict=False)
            model.classifier = torch.nn.Linear(1920, output_neurons)

        elif self.baseline_model.startswith('UKBBINITIAL'):
            # Get best epoch
            outcome = 'RVEF' if 'RVEF' in self.outcome else 'RVEDVI'

            results_file = f'Results/UKBBINITIAL_{outcome}_{self.class_reg_string}_Results.pickle'
            df_results = pd.read_pickle(results_file)
            df_results = df_results.query('eval_mode == "test"').reset_index(drop=True)

            if self.is_classification:
                best_epoch = df_results.iloc[df_results['metric'].idxmax()]['epoch']
            else:
                best_epoch = df_results.iloc[df_results['metric'].idxmin()]['epoch']

            # Load baseline model
            # Get model path
            baseline_model_path = f'Models/{self.baseline_model}/{self.class_reg_string}/model_{best_epoch}.pt'
            print(f'Loading baseline model from {baseline_model_path}')

            # Load state dict and remove module. from keys
            model = torchvision.models.densenet201(weights=None)
            state_dict = self.load_existing_model(baseline_model_path)
            model.classifier = torch.nn.Linear(1920, output_neurons)
            model.load_state_dict(state_dict, strict=False)

        return model

    def gaping_maw(self, train_dataloader, eval_dataloaders):
        # Classification / Regression setup
        output_neurons = 2 if self.is_classification else 1
        criterion = torch.nn.CrossEntropyLoss() if self.is_classification else torch.nn.MSELoss()
        label_dtype = torch.long if self.is_classification else torch.float32

        # Create model
        model = self.create_model(output_neurons)

        # Move to GPU and Dataparallel
        if not Config.single_gpu:
            model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        model = model.cuda()

        # Define optimizer, scaler and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)
        scaler = torch.cuda.amp.GradScaler()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=Config.ft_epochs,
            steps_per_epoch=len(train_dataloader))

        # Save results here - outside of the loop
        all_results = []

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
                    loss = criterion(outputs.squeeze(), labels.to(label_dtype))
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
                            testing_loss = criterion(outputs.squeeze(), labels.to(label_dtype))
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
                    df_pred['pred'] = df_pred['pred'].str[1]  # Crossentropy loss returns [0, 1] for each class
                    metric = metrics.roc_auc_score(df_pred['label'], df_pred['pred'])
                else:
                    metric = metrics.mean_absolute_error(df_pred['label'], df_pred['pred'])

                all_results.append([desc, epoch, training_epoch_loss, testing_epoch_loss, metric])
                print(f'Epoch: {epoch}, Training Loss: {training_epoch_loss}, {desc.upper()} Loss: {testing_epoch_loss}, METRIC: {metric}')

                # Save predictions
                outfile_name = f'{self.outdir}/epoch_{epoch}.pickle'
                df_pred.to_pickle(outfile_name)

        df_results = pd.DataFrame(all_results, columns=['eval_mode', 'epoch', 'training_loss', 'testing_loss', 'metric'])
        df_results.to_pickle(f'Results/{self.outcome}_{self.class_reg_string}_Results.pickle')

    def create_dataloaders(self):
        # Assumes all files exist
        if self.outcome.startswith('UKBBINITIAL'):
            df = pd.read_pickle(Config.file_ukbb_oc_final)
            df['FILENAME'] = df['FILENAME'].apply(lambda x: os.path.join(Config.dir_ukbb_ecgs, x))
        else:
            df = pd.read_pickle(Config.file_sinai_oc_final)
            df['FILENAME'] = df['FILENAME'].apply(lambda x: os.path.join(Config.dir_sinai_ecgs, x))

        # Cleanup: RVEF between 10 and 80
        df = df.query('10 < RVEF < 80')
        df = df.reset_index(drop=True)

        if 'RVEF' in self.outcome:
            if self.is_classification:
                df['label'] = df['RVEF'].apply(lambda x: 1 if x <= Config.rvef_classification else 0)
            else:
                df['label'] = df['RVEF']

        if 'RVEDVI' in self.outcome:
            if self.is_classification:
                df['label'] = df['RVEDVI'].apply(lambda x: 1 if x <= Config.rvedv_classification else 0)
            else:
                df['label'] = df['RVEDVI']

        # Restrict to required columns
        df = df[['FILENAME', 'label', 'MRN']]
        df = df.dropna()

        # Group shuffle split
        gss = model_selection.GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=Config.random_state)
        train_idx, test_idx = next(gss.split(df, groups=df['MRN']))

        train = df.iloc[train_idx]
        train, validation = model_selection.train_test_split(
            train, test_size=0.05, random_state=Config.random_state)
        test = df.iloc[test_idx]

        # Assert for length
        assert len(train) + len(validation) + len(test) == len(df), 'Length mismatch'

        if Config.debug:
            train = train.sample(100)
            validation = validation.sample(100)
            test = test.sample(100)

        print('Train', train.shape, 'Validation', validation.shape, 'Test', test.shape)

        # Create dataloaders
        train_dataset = ECGDataset(train)
        val_dataset = ECGDataset(validation)
        test_dataset = ECGDataset(test)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=Config.batch_size, shuffle=True,
            num_workers=Config.num_workers, pin_memory=True)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=Config.batch_size, shuffle=False,
            num_workers=Config.num_workers, pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=Config.batch_size, shuffle=False,
            num_workers=Config.num_workers, pin_memory=True)

        return train_dataloader, val_dataloader, test_dataloader

    def hammer_time(self):
        # Housekeeping
        self.outdir = f'Results/{self.outcome}/{self.class_reg_string}'
        self.modeldir = f'Models/{self.outcome}/{self.class_reg_string}'
        
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(self.modeldir, exist_ok=True)

        # Proceed as usual
        train_dataloader, val_dataloader, test_dataloader = self.create_dataloaders()

        eval_dataloaders = {
            'validation': val_dataloader,
            'test': test_dataloader}

        self.gaping_maw(train_dataloader, eval_dataloaders)
