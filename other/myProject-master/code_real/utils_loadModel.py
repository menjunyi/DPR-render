import importlib
import torch

def loadModels(args):
    coarseModel = None
    fineModel = None
    detailModel = None
    if args.coarseModel:
        coarseModule = importlib.import_module(args.coarseModel)
        coarseModel = coarseModule.HourglassNet(args.coarse_nChannel)
        if args.coarseModel_load:
            #coarseModel.load_state_dict(torch.load(args.coarseModel_load))
            coarseModel = torch.load(args.coarseModel_load)
    
    if args.fineModel:
        fineModule = importlib.import_module(args.fineModel)
        fineModel = fineModule.DRN(args.fine_albedoChannel, 
                args.fine_normalChannel, args.fine_lightingChannel)
        if args.fineModel_load:
            #fineModel.load_state_dict(torch.load(args.fineModel_load))
            fineModel = torch.load(args.fineModel_load)
    
    if args.detailModel:
        detailModule = importlib.import_module(args.detailModel)
        detailModel = detailModule.DRN(args.detail_albedoChannel,
                args.detail_normalChannel, args.detail_lightingChannel)
        if args.detailModel_load:
            #detailModel.load_state_dict(torch.load(args.detailModel_load))
            detailModel = torch.load(args.detailModel_load)
    return coarseModel, fineModel, detailModel


def loadDiscriminator(args):
    netD_coarse = None
    netD_fine = None
    netD_detail = None
    if args.D_coarseModel:
        coarseModule = importlib.import_module(args.D_coarseModel)
        netD_coarse = coarseModule.Discriminator_coarse()
        if args.D_coarseModel_load:
            netD_coarse = torch.load(args.D_coarseModel_load)

    if args.D_fineModel:
        fineModule = importlib.import_module(args.D_fineModel)
        netD_fine = fineModule.Discriminator_fine()
        if args.D_fineModel_load:
            netD_fine = torch.load(args.D_fineModel_load)

    if args.D_detailModel:
        detailModule = importlib.import_module(args.D_detailModel)
        netD_detail = detailModule.Discriminator_detail()
        if args.D_detailModel_load:
            netD_detail = torch.load(args.D_detailModel_load)
    return netD_coarse, netD_fine, netD_detail
    
