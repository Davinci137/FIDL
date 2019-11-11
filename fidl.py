import click
import os 
import libfidl


@click.command()
@click.option('--adduser', type=libfidl.Username(), help = '[username] Give a new person access to the room. This retrains the AI and provides access to the Website.')
@click.option('--access', type = (str,str), default = (None, None), help = '[username] [y/n] Change the Access of a user to the room and the website.')
@click.option('--passwd', type=libfidl.Username(new=False), help = '[username] Change the password of an existing user')
def main(adduser, access, passwd):
    if adduser:
        path = click.prompt('Path to new users pictures', type=click.Path(exists=True, file_okay= False, dir_okay= True))

        hashed_password = libfidl.hash(click.prompt('Password', hide_input=True, confirmation_prompt=True), 'sha512')
        if click.confirm('Are you sure to grant {} access?'.format(adduser)):
            click.echo(click.style('Granting access', fg= 'green'))

            #Save user to properties
            props['user'][adduser] = {'password': hashed_password, 'access': True}
            libfidl.save_properties(props)
            
            #retrain Model
            libfidl.retrain_model(adduser,path)
            click.echo(click.style('Done. User added!', fg= 'green'))
        else:
            click.echo(click.style('Abort', fg= 'red'))

    if not None in access:
        username = access[0]
        new_access = access[1].lower()

        #check if the user exists
        if not libfidl.user_exist(username):
            click.echo(click.style('User {} does not exist'.format(username), fg = 'red'))
            return  
            
        #check if new_access has the right format
        if not new_access in {'y','n'}:
            click.echo(click.style('Abort. Access quantifier must be y or n', fg='red'))
            return

        if click.confirm('Are you sure to {} {} access?'.format('grant' if new_access == 'y' else 'deny',username)):
            click.echo(click.style('Changing access', fg= 'green'))
            #changing access
            props['user'][username]['access'] = True if new_access == 'y' else False
            #saving properties
            libfidl.save_properties(props)
        else:
            click.echo(click.style('Abort', fg= 'red'))

    if passwd:
        #get new password
        hashed_password = libfidl.hash(click.prompt('New password', hide_input=True, confirmation_prompt=True), 'sha512')
        #save new password
        props['user'][passwd]['password'] = hashed_password
        libfidl.save_properties(props) 

    if passwd == None and access == (None,None) and adduser == None: 
        #Run facial recognition
        libfidl.run_recognize()

if __name__ == '__main__':
    props = libfidl.load_properties()
    main()