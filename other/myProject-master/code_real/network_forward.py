import torch.nn.functional as F
import torch
from torch.autograd import Variable

class networkForward_basic(object):
    '''
        farward network to get all possible results
    '''
    def __init__(self, model, shadingModel, imageSize, stage):
        '''
            model: network model
            state: can choose 1, 2, 3:
                   1: coarse level
                   2: fine level
                   3: detail level
        '''
        self.model = model
        self.shadingModel = shadingModel
        self.stage = stage
        self.imageSize = imageSize
        if self.stage == 1:
            self.forward = self.forward_coarse
        elif self.stage == 2:
            self.forward = self.forward_fine
        elif self.stage == 3:
            self.forward = self.forward_detail

    def forward_coarse(self, data):
        '''
            forward for coarse level
        '''
        if data.shape[3] != self.imageSize:
            data = F.upsample(data, size=[self.imageSize, self.imageSize], mode='bilinear')
        albedo, normal, lighting = self.model(data)
        shading = self.shadingModel(normal, lighting)
        shading = F.relu(shading)
        output = {}
        output['albedo'] = albedo
        output['normal'] = normal
        output['shading'] = shading
        output['lighting'] = lighting
        #return albedo, normal, shading, lighting
        return output

    def forward_fine(self, data, coarse_data):
        '''
            forward for fine level
        '''
        if data.shape[3] != self.imageSize:
            data = F.upsample(data, size=[self.imageSize, self.imageSize], mode='bilinear')
        coarse_albedo = coarse_data['albedo']
        coarse_normal = coarse_data['normal']
        coarse_shading = coarse_data['shading']
        coarse_lighting = coarse_data['lighting']

        # upsample to the correct size 
        coarse_albedo = F.upsample(coarse_albedo, 
                size=[self.imageSize, self.imageSize], mode='bilinear')
        coarse_normal = F.upsample(coarse_normal, 
                size=[self.imageSize, self.imageSize], mode='bilinear')
        coarse_shading = F.upsample(coarse_shading, 
                size=[self.imageSize, self.imageSize], mode='bilinear')

        # NOTE: we have a bug in coarse network for lighting, correct it
        coarse_lighting = Variable(coarse_lighting[:,0:27].data).float()
        coarse_lighting = coarse_lighting.unsqueeze(-1)
        coarse_lighting = coarse_lighting.unsqueeze(-1)

        # prepare inputs
        inputs_albedo = torch.cat((data, coarse_albedo), dim=1)
        inputs_normal = torch.cat((data, coarse_normal), dim=1)
        inputs_lighting = torch.cat((data, coarse_albedo, coarse_normal, coarse_shading), dim=1)

        # predict residual
        output_albedo, output_normal, output_lighting = \
                self.model(inputs_albedo, inputs_normal, inputs_lighting)

        lighting = output_lighting + coarse_lighting.expand(-1,-1, self.imageSize, self.imageSize)
        albedo = output_albedo + coarse_albedo
        normal = F.normalize(output_normal + coarse_normal, p=2, dim=1)

        # get shading
        shading = self.shadingModel(F.normalize(normal, p=2,dim=1), lighting)

        output={}
        output['albedo'] = albedo
        output['normal'] = normal
        output['shading'] = shading
        output['lighting'] = lighting
        return output

    def forward_detail(self, data, fine_data):

        '''
            forward for detail level
        '''
        if data.shape[3] != self.imageSize:
            data = F.upsample(data, size=[self.imageSize, self.imageSize], mode='bilinear')
        fine_albedo = fine_data['albedo']
        fine_normal = fine_data['normal']
        fine_shading = fine_data['shading']
        fine_lighting = fine_data['lighting']

        # upsample to correct size 
        fine_albedo = F.upsample(fine_albedo, 
                size=[self.imageSize, self.imageSize], mode='bilinear')
        fine_normal = F.upsample(fine_normal, 
                size=[self.imageSize, self.imageSize], mode='bilinear')
        fine_shading = F.upsample(fine_shading, 
                size=[self.imageSize, self.imageSize], mode='bilinear')
        fine_lighting = F.upsample(fine_lighting, 
                size=[self.imageSize, self.imageSize], mode='bilinear')

        # prepare inputs
        inputs_albedo = torch.cat((data, fine_albedo), dim=1)
        inputs_normal = torch.cat((data, fine_normal), dim=1)
        inputs_lighting = torch.cat((data, fine_albedo, fine_normal, fine_shading), dim=1)

        # predict residual
        output_albedo, output_normal, output_lighting = \
                self.model(inputs_albedo, inputs_normal, inputs_lighting)

        lighting = output_lighting + fine_lighting
        albedo = output_albedo + fine_albedo
        normal = F.normalize(output_normal + fine_normal, p=2, dim=1)

        # get shading
        shading = self.shadingModel(F.normalize(normal, p=2,dim=1), lighting)
        output={}
        output['albedo'] = albedo
        output['normal'] = normal
        output['shading'] = shading
        output['lighting'] = lighting
        return output
