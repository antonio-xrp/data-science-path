CREATE USER 'antonio'@'localhost' IDENTIFIED BY 'qwerty';

# agregar todos los privilegios a un usuario
GRANT ALL PRIVILEGES ON antonio_db TO 'antonio'@'localhost';

#Como buena práctica despues de asignar privilegios correr el código
FLUSH PRIVILEGES;

#Para ver los privilegios asignados a un usuario
SHOW GRANTS FOR 'antonio'@'localhost';

#Para revocar todos los privilegios asignados a un usuario
REVOKE ALL, GRANT OPTION FROM 'antonio'@'localhost';

#para eliminar un usuario
DROP USER 'antonio'@'localhost';