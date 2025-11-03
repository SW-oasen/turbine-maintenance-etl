{% set cols = adapter.get_columns_in_relation( source('main','cycles_raw') ) %}

{% set sensor_cols = [] %}
{% for col in cols %}
  {# Prüfe Prefix via Slicing: die ersten 6 Zeichen == 'sensor' #}
  {% if (col.name | lower)[:6] == 'sensor' %}
    {% do sensor_cols.append(col.name) %}
  {% endif %}
{% endfor %}

select
  cast(dataset as text)        as dataset,
  cast(unit_nr as integer)     as unit_nr,
  cast(time_cycles as integer) as time_cycles,
  cast(setting1 as real)       as setting1,
  cast(setting2 as real)       as setting2,
  cast(setting3 as real)       as setting3,
  {% for c in sensor_cols -%}
    cast({{ c }} as real) as {{ c }}{% if not loop.last %},{% endif %}
  {%- endfor %}
from {{ source('main','cycles_raw') }}
