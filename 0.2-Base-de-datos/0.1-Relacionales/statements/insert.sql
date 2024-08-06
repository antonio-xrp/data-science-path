
# Para agregar valores a una tabla, primero se especifica las columnas a agregar y luego los valores
INSERT INTO usuarios (apellidos, edad, nombre) VALUES ("Palacios","15","Antonio");

# otra forma de hacer lo mismo
INSERT INTO usuarios SET nombre='rosa', apellidos='campos', edad=22;

# Para agregar varios valores a una tabla
INSERT INTO usuarios (nombre, apellidos, correo, edad) VALUES
	("duke", "rojas", "duke@gmail.com", 1),
    ("black", "palacios", "goma@gmail.com", 4);

# Para mostrar los datos de una tabla
SELECT * FROM usuarios;

# Para mostrar solo los nombres y edades de los usuarios
SELECT nombre, edad, usuario_id FROM usuarios;

# Para contar cuantos registros hay en una tabla
SELECT COUNT(*) FROM usuarios;

# Para darle un alias a la columna de la tabla
SELECT COUNT(*) AS total_usuarios FROM usuarios;

# Para seleccionar solo los usuarios que cumplan con una condición
SELECT * FROM usuarios WHERE nombre = 'Antonio';

SELECT * FROM usuarios WHERE nombre in ("Antonio", "duke", "black");

# Para seleccionar solo los usuarios que cumplan con una condició, en este caso todos los apellidos que empiecen con 'p'
SELECT * FROM usuarios where apellidos LIKE 'p%';

SELECT * FROM usuarios where correo LIKE '%@gmail.com';

SELECT * FROM usuarios where correo LIKE '%it%';

# PARA LO INVERSO DE LIKE
SELECT * FROM usuarios where apellidos NOT LIKE 'p%';

SELECT * FROM usuarios where correo NOT LIKE '%@gmail.com';

# Para seleccionar solo los usuarios que cumplan con una condición, en este caso todos los usuarios que tengan una edad diferente a 15
SELECT * FROM usuarios WHERE edad != 15;
#otra manera de hacer lo mismo
SELECT * FROM usuarios WHERE edad <> 15;


SELECT * FROM usuarios WHERE edad = 15;
SELECT * FROM usuarios WHERE edad > 1;
SELECT * FROM usuarios WHERE edad >= 1;
SELECT * FROM usuarios WHERE edad <= 15;
