{% set cols = adapter.get_columns_in_relation( source('main','cycles_raw') ) %}

{% set sensor_cols = [] %}
{% for col in cols %}
  {% if (col.name | lower)[:6] == 'sensor' %}
    {% do sensor_cols.append(col.name) %}
  {% endif %}
{% endfor %}

with base as (
  select * from {{ ref('stg_cycles_raw') }}
),
feat as (
  select
    -- dataset,
    -- unit_nr,
    -- time_cycles,
    -- max(time_cycles) over (partition by unit_nr) - time_cycles as rul

    cast(dataset as text)          as dataset,       -- hart typisieren
    cast(unit_nr as integer)       as unit_nr,
    cast(time_cycles as integer)   as time_cycles,
    cast(
      max(time_cycles) over (partition by unit_nr) - time_cycles
      as integer
    )                              as rul


    {% for c in sensor_cols %}
    , avg({{ c }}) over (
        partition by unit_nr
        order by time_cycles
        rows between 4 preceding and current row
      ) as mean5_{{ c }}
    {% endfor %}

    {% for c in sensor_cols %}
    , avg({{ c }}) over (
        partition by unit_nr
        order by time_cycles
        rows between 19 preceding and current row
      ) as mean20_{{ c }}
    {% endfor %}

    {% for c in sensor_cols %}
    , ( {{ c }} - lag({{ c }}) over (partition by unit_nr order by time_cycles) ) as d_{{ c }}
    {% endfor %}
  from base
)
select * from feat
