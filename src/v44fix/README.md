# GeoCPU v44fix source snapshot

This directory contains the Python source and metadata from the existing
`GeoCPU_v44fix_bundle.zip` in the user's private Drive folder. It is the
**unmodified baseline** for the repair review. It retains the v43 CPU/ALU and
search paths and adds Python as another tool. It is distinct from the earlier
pure-Python v44 branch.

Training data, evaluation bodies, model weights, checkpoints and credentials
are excluded. The baseline is preserved so a repair pull request can show
the exact source changes. This snapshot has not established post-training
quality or a safe environment for executing public Python examples.

The repair branch is a code-review artifact. Its release manifest sets
`training_ready=false`. It must not be used as a Colab training launcher until
public Python examples are verified in a credential-free isolated environment
and the Python execution path is isolated from Drive credentials. The original
v43 checkpoint and source ZIP remain outside this repository.
