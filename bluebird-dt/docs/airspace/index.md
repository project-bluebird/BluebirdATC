# Provided Airspace

Users can choose between various simulated *Scenarios*, many of which can take place in a chosen *Airspace*.   

The Airspaces included in the `bluebird-dt` digital twin at the moment are:

### I-sector

This is a simple linear sector, with only two entry/exit windows at the North and South ends.

```python
from bluebird_dt.airspace_generator import SectorI
airspace, routes = SectorI(
                        width=20, 
                        height=60, 
                        fl_limits=(200, 300)
                    ).generate_airspace()
```

<div style="text-align: center;">
<img src="../images/sector_i.png" width="200" position="center">
</div>

The API reference for the SectorI constructor is available [in the source reference](../source.md#bluebird_dt.airspace_generator.SectorI)

### Y-sector

This Y-shaped sector can have aircraft coming from any of three directions, and converging on the fix in the centre.

```python
from bluebird_dt.airspace_generator import SectorY
airspace, routes = SectorY(
                        width=20, 
                        height=60, 
                        fl_limits=(200, 300),
                        alpha=52.5,
                    ).generate_airspace()
```

<div style="text-align: center;">
<img src="../images/sector_y.png" width="400" position="center">
</div>

The API reference for the SectorY constructor is available [in the source reference](../source.md#bluebird_dt.airspace_generator.SectorY)

### X-sector

The X-sector is essentially two I-sectors at 90 degree angle to one another, crossing over in the centre.  There are numerous possibilities for conflicts arising from aircraft entering from different legs of the sector.

```python
from bluebird_dt.airspace_generator import SectorX
airspace, routes = SectorX(
                        width=20, 
                        height=60, 
                        fl_limits=(200, 300),
                        alpha=52.5,
                    ).generate_airspace()
```

<div style="text-align: center;">
<img src="../images/sector_x.png" width="500" position="center">
</div>


The API reference for the SectorX constructor is available [in the source reference](../source.md#bluebird_dt.airspace_generator.SectorX)

### Xplus-sector

The Xplus-sector is similar to the X-sector, but with an asymmetric cutout, meaning there is a wider region of allowed airspace, and more possible routes.

```python
from bluebird_dt.airspace_generator import SectorXPlus
airspace, routes = SectorXPlus().generate_airspace()
```

<div style="text-align: center;">
<img src="../images/sector_xplus.png" width="500" position="center">
</div>

The API reference for the SectorXPlus constructor is available [in the source reference](../source.md#bluebird_dt.airspace_generator.SectorXPlus)

### Springfield 

The Springfield sector, with its surrounding sectors, is a synthetic airspace that is designed to test agents on many of the challenging situations that could arise in real ATC.

<div style="text-align: center;">
    <img src="../images/sector_springfield.png" width="600" position="center">
</div>

The Springfield airspace is initialised using the Springfield scenario manager described [here](../scenarios/index.md#springfield)
