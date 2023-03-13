Blocks
======


Encoders
--------

.. currentmodule:: tsl.nn.blocks.encoders

.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.blocks.encoders.enc_classes %}
     {{ cls }}
   {% endfor %}


Recurrent Encoders
++++++++++++++++++

.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.blocks.encoders.rnn_classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.blocks.encoders
    :members:


Decoders
--------

.. currentmodule:: tsl.nn.blocks.decoders
.. autosummary::
   :nosignatures:
   {% for cls in tsl.nn.blocks.decoders.classes %}
     {{ cls }}
   {% endfor %}

.. automodule:: tsl.nn.blocks.decoders
    :members:
