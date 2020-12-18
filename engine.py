import torch
import torch.nn as nn
import numpy as np

from dataset.voc_eval import evaluate


def train(epoch, model, criterion, optimizer, data_loader_train, device, args, writer, training_loss):
    total_loss = 0

    model.train()
    for i, (imgs, targets) in enumerate(data_loader_train):
        imgs, targets = imgs.to(device), targets.to(device)

        pred = model(imgs)
        loss = criterion(pred, targets)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        if args.max_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        if (i+1) % args.log_every == 0 or i == 0 or (i+1) == len(data_loader_train):
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Average Loss: %.4f'
                  % (epoch + 1, args.epoch, i+1, len(data_loader_train), loss.item(), total_loss/(i+1)))
            if (i+1) == len(data_loader_train):
                training_loss.append(total_loss/(i+1))
    writer.add_scalar('Training Loss', total_loss/len(data_loader_train), epoch+1)


def eval(epoch, model, criterion, data_loader_test, test_file, device, args, writer, testing_loss, eval_map):
    best_test_loss = np.inf

    with torch.no_grad():
        total_loss = 0

        model.eval()
        for i, (imgs, targets) in enumerate(data_loader_test):
            imgs, targets = imgs.to(device), targets.to(device)

            pred = model(imgs)
            loss = criterion(pred, targets)
            total_loss += loss.item()
        total_loss /= len(data_loader_test)
        test_aps = evaluate(model, test_dataset_file=test_file)
        print("Epoch [%d/%d], Testing Loss: %.4f" % (epoch+1, args.epoch, total_loss))
        writer.add_scalar("Test mAP", np.mean(test_aps), epoch + 1)
        writer.add_scalar('Testing Loss', total_loss, epoch + 1)
        testing_loss.append(total_loss)
        eval_map.append(np.mean(test_aps))

    if best_test_loss > total_loss:
        best_test_loss = total_loss
        torch.save(model.state_dict(), "best_model.pth")
