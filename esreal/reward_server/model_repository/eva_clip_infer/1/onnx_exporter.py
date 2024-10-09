# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os

import torch
import torchvision.models as models


class MergeModel(torch.nn.Module):
    def __init__(self, model_A, model_B):
        super(MergeModel, self).__init__()
        self.model_A = model_A
        self.model_B = model_B

    def forward(self, x):
        return self.model_B(self.model_A(x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", default="model.onnx")
    args = parser.parse_args()

    from blip2 import init_vision_encoder

    visual_encoder, ln_vision = init_vision_encoder(precision="fp32")
    visual_encoder.to("cuda")
    ln_vision.to("cuda")
    model = MergeModel(visual_encoder, ln_vision)
    model.to("cuda")
    model.eval()

    # resnet50 = models.resnet50(pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32, device="cuda")
    # resnet50 = resnet50.eval()

    torch.onnx.export(
        model,
        dummy_input,
        args.save,
        export_params=True,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print("Saved {}".format(args.save))
