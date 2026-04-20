# Fixes and improvements

This note tracks fixes and improvements we are rolling into the released
code and models.

## 2026-04-16

The changes below make the released models more robust for in-the-wild
use. Please use [`interactvlm-3d-hcontact-damon-fix`](https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=interactvlm-3d-hcontact-damon-fix.zip) for DAMON 3D
human-contact evaluation from this point on; the numbers below are
produced by this model. An arXiv update with the new numbers and
re-releases of the other Model Zoo entries are on the way.

- **Validation inference mode.** `evaluate.py` now defaults to autoregressive
  `generate` mode. The previous `forward` path (from
  [LISA](https://github.com/dvlab-research/LISA)) used teacher forcing, where
  body-part tokens were included in the prompt while producing `[SEG]`. In
  `generate` mode, the model must generate body parts before `[SEG]`,
  matching test-time usage. Thanks to
  [Ha Linh Nguyen](https://www.comp.nus.edu.sg/~hlinhn/) for first reporting this issue. 

- **Body-part dropout during training (`hC_body_part_dropout_prob`).**
  Training still uses teacher forcing, which creates a mismatch with
  `generate` inference. To reduce this gap, we drop body-part tokens with
  probability `p` by switching from the `parts` template to the `simple`
  template. This forces the model to predict masks from visual evidence
  alone in a subset of updates, improving generalization.

- **Improved per-view GT contact masks (`mv2`).**
  `generate_damon_human_mask.py` now supports `--min_vertices 2`
  (previously 3), and DAMON training uses the `4MV-Z_Vitru_mv2` view set.
  A contact triangle is retained if at least two vertices project inside
  the silhouette, instead of all three. This stabilizes boundary regions.
  The overall gain is modest; masks from the previous version remain
  usable.

- **3D contact predictor.** `HumanContact3DPredictor` now aggregates views
  with a soft sigmoid and barycentric-weighted scatter, restoring gradient
  flow through the 2D→3D step. The previous hard-threshold version was
  effectively detached.

- **Loss cleanups.** `compute_dice_loss` no longer returns early on
  empty-GT views, and `HumanContact3DLoss` clamps its inputs before BCE.

- **Binary contact metric threshold.** `get_damon_binary_contact` now
  thresholds predictions at `0.5` before forming the per-image union.

## Updated DAMON numbers

Two models are released with this update, both evaluated on the full DAMON
test split (1370 samples) with `inference_type=generate` and threshold `0.5`:

- [`interactvlm-3d-hcontact-damon-fix`](https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=interactvlm-3d-hcontact-damon-fix.zip) — `parts` answer template (`"The contacting body parts are {body_parts}, and the contact region is [SEG]."`). Use this for the strongest numbers.
- [`interactvlm-3d-hcontact-damon-noParts`](https://download.is.tue.mpg.de/download.php?domain=interactvlm&sfile=interactvlm-3d-hcontact-damon-noParts.zip) — `simple` answer template (`"Sure, [SEG]."`). 

### Binary contact (per-image)

| Model | F1 | Precision | Recall |
|---|---|---|---|
| `interactvlm-3d-hcontact-damon-fix` | **70.32** | 67.91 | 78.91 |
| `interactvlm-3d-hcontact-damon-noParts` | **64.46** | 67.34 | 68.31 |

### Semantic contact (per-object)

<table>
  <thead>
    <tr>
      <th rowspan="2">Category</th>
      <th rowspan="2"># samples</th>
      <th colspan="3">with body parts (<code>damon-fix</code>)</th>
      <th colspan="3">without body parts (<code>damon-noParts</code>)</th>
    </tr>
    <tr>
      <th>F1</th><th>Precision</th><th>Recall</th>
      <th>F1</th><th>Precision</th><th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>transport</td><td>87</td><td>72.25</td><td>67.84</td><td>82.36</td><td>70.35</td><td>70.49</td><td>76.97</td></tr>
    <tr><td>sports</td><td>305</td><td>72.44</td><td>71.20</td><td>81.47</td><td>66.64</td><td>69.16</td><td>71.84</td></tr>
    <tr><td>kitchen</td><td>38</td><td>65.35</td><td>62.12</td><td>75.31</td><td>57.32</td><td>57.04</td><td>64.85</td></tr>
    <tr><td>food</td><td>32</td><td>62.05</td><td>57.61</td><td>76.67</td><td>56.08</td><td>52.35</td><td>70.82</td></tr>
    <tr><td>accessory</td><td>47</td><td>60.34</td><td>56.15</td><td>70.05</td><td>46.45</td><td>47.65</td><td>51.32</td></tr>
    <tr><td>furniture</td><td>146</td><td>58.90</td><td>54.04</td><td>72.87</td><td>45.04</td><td>57.59</td><td>42.78</td></tr>
    <tr><td>everyday-objects</td><td>174</td><td>54.98</td><td>53.62</td><td>63.42</td><td>52.17</td><td>50.45</td><td>64.20</td></tr>
    <tr><td>supporting</td><td>541</td><td>68.84</td><td>67.43</td><td>76.78</td><td>65.23</td><td>67.14</td><td>70.17</td></tr>
    <tr><td><b>DAMON (weighted)</b></td><td><b>1370</b></td><td><b>66.49</b></td><td><b>64.35</b></td><td><b>75.79</b></td><td><b>60.98</b></td><td><b>63.37</b></td><td><b>66.52</b></td></tr>
  </tbody>
</table>

Please use these numbers for any comparison against InteractVLM on DAMON.
