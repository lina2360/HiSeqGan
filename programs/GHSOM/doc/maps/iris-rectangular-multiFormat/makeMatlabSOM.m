data = som_read_data('iris.sompak.data');
map = som_make(data, 'msize', [12 18], 'lattice', 'rect', 'algorithm', 'seq', 'training', 'long');
som_write_cod(map, 'iris.sompak.wgt');
somvis_gui(map, data);
