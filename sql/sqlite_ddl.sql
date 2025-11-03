PRAGMA foreign_keys = ON;
DROP TABLE IF EXISTS cycles_raw;
CREATE TABLE cycles_raw (
  dataset TEXT, unit_nr INTEGER, time_cycles INTEGER,
  setting1 REAL, setting2 REAL, setting3 REAL,
  sensor1 REAL, sensor2 REAL, sensor3 REAL, sensor4 REAL, sensor5 REAL,
  sensor6 REAL, sensor7 REAL, sensor8 REAL, sensor9 REAL, sensor10 REAL,
  sensor11 REAL, sensor12 REAL, sensor13 REAL, sensor14 REAL, sensor15 REAL,
  sensor16 REAL, sensor17 REAL, sensor18 REAL, sensor19 REAL, sensor20 REAL, sensor21 REAL,
  sensor22 REAL, sensor23 REAL, sensor24 REAL, sensor25 REAL, sensor26 REAL
  PRIMARY KEY (dataset, unit_nr, time_cycles)
);

DROP TABLE IF EXISTS cycles_features;
CREATE TABLE cycles_features (
  dataset TEXT, unit_nr INTEGER, time_cycles INTEGER, rul INTEGER,
  mean5_sensor2 REAL, mean5_sensor3 REAL, mean5_sensor4 REAL,
  mean20_sensor2 REAL, mean20_sensor3 REAL, mean20_sensor4 REAL,
  d_sensor2 REAL, d_sensor3 REAL, d_sensor4 REAL,
  z_sensor2 REAL, z_sensor3 REAL, z_sensor4 REAL,
  PRIMARY KEY (dataset, unit_nr, time_cycles)
);

DROP TABLE IF EXISTS units_summary;
CREATE TABLE units_summary (
  dataset TEXT, unit_nr INTEGER, cycles_min INTEGER, cycles_max INTEGER, cycles_count INTEGER,
  PRIMARY KEY (dataset, unit_nr)
);

CREATE INDEX IF NOT EXISTS idx_cycles_raw_unit ON cycles_raw(dataset, unit_nr);
CREATE INDEX IF NOT EXISTS idx_cycles_features_unit ON cycles_features(dataset, unit_nr);