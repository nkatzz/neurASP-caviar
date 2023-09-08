
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(mws1gt_1,155,214). coords(mws1gt_1,131,131,214).  visible(mws1gt_1,214). happensAt(walking(mws1gt_1),214). orientation(mws1gt_5,161,214). coords(mws1gt_5,45,113,214).  visible(mws1gt_5,214). happensAt(walking(mws1gt_5),214).
orientation(mws1gt_1,155,215). coords(mws1gt_1,130,131,215).  visible(mws1gt_1,215). happensAt(walking(mws1gt_1),215). orientation(mws1gt_5,161,215). coords(mws1gt_5,46,116,215).  visible(mws1gt_5,215). happensAt(walking(mws1gt_5),215).
orientation(mws1gt_1,155,216). coords(mws1gt_1,128,132,216).  visible(mws1gt_1,216). happensAt(walking(mws1gt_1),216). orientation(mws1gt_5,161,216). coords(mws1gt_5,46,117,216).  visible(mws1gt_5,216). happensAt(walking(mws1gt_5),216).
orientation(mws1gt_1,155,217). coords(mws1gt_1,127,132,217).  visible(mws1gt_1,217). happensAt(walking(mws1gt_1),217). orientation(mws1gt_5,161,217). coords(mws1gt_5,46,117,217).  visible(mws1gt_5,217). happensAt(walking(mws1gt_5),217).
orientation(mws1gt_1,155,218). coords(mws1gt_1,126,132,218).  visible(mws1gt_1,218). happensAt(walking(mws1gt_1),218). orientation(mws1gt_5,161,218). coords(mws1gt_5,46,118,218).  visible(mws1gt_5,218). happensAt(walking(mws1gt_5),218).
orientation(mws1gt_1,155,219). coords(mws1gt_1,126,132,219).  visible(mws1gt_1,219). happensAt(walking(mws1gt_1),219). orientation(mws1gt_5,161,219). coords(mws1gt_5,46,118,219).  visible(mws1gt_5,219). happensAt(walking(mws1gt_5),219).
orientation(mws1gt_1,155,220). coords(mws1gt_1,126,132,220).  visible(mws1gt_1,220). happensAt(walking(mws1gt_1),220). orientation(mws1gt_5,161,220). coords(mws1gt_5,46,119,220).  visible(mws1gt_5,220). happensAt(walking(mws1gt_5),220).
orientation(mws1gt_1,155,221). coords(mws1gt_1,126,131,221).  visible(mws1gt_1,221). happensAt(walking(mws1gt_1),221). orientation(mws1gt_5,161,221). coords(mws1gt_5,46,120,221).  visible(mws1gt_5,221). happensAt(walking(mws1gt_5),221).
orientation(mws1gt_1,155,222). coords(mws1gt_1,125,131,222).  visible(mws1gt_1,222). happensAt(walking(mws1gt_1),222). orientation(mws1gt_5,161,222). coords(mws1gt_5,46,120,222).  visible(mws1gt_5,222). happensAt(walking(mws1gt_5),222).
orientation(mws1gt_1,155,223). coords(mws1gt_1,125,130,223).  visible(mws1gt_1,223). happensAt(walking(mws1gt_1),223). orientation(mws1gt_5,161,223). coords(mws1gt_5,46,120,223).  visible(mws1gt_5,223). happensAt(walking(mws1gt_5),223).
orientation(mws1gt_1,155,224). coords(mws1gt_1,125,130,224).  visible(mws1gt_1,224). happensAt(walking(mws1gt_1),224). orientation(mws1gt_5,161,224). coords(mws1gt_5,46,120,224).  visible(mws1gt_5,224). happensAt(walking(mws1gt_5),224).
orientation(mws1gt_1,155,225). coords(mws1gt_1,125,130,225).  visible(mws1gt_1,225). happensAt(walking(mws1gt_1),225). orientation(mws1gt_5,161,225). coords(mws1gt_5,46,120,225).  visible(mws1gt_5,225). happensAt(walking(mws1gt_5),225).
orientation(mws1gt_1,155,226). coords(mws1gt_1,125,130,226).  visible(mws1gt_1,226). happensAt(walking(mws1gt_1),226). orientation(mws1gt_5,161,226). coords(mws1gt_5,45,120,226).  visible(mws1gt_5,226). happensAt(walking(mws1gt_5),226).
orientation(mws1gt_1,155,227). coords(mws1gt_1,124,130,227).  visible(mws1gt_1,227). happensAt(walking(mws1gt_1),227). orientation(mws1gt_5,161,227). coords(mws1gt_5,45,121,227).  visible(mws1gt_5,227). happensAt(walking(mws1gt_5),227).
orientation(mws1gt_1,155,228). coords(mws1gt_1,123,129,228).  visible(mws1gt_1,228). happensAt(walking(mws1gt_1),228). orientation(mws1gt_5,161,228). coords(mws1gt_5,45,124,228).  visible(mws1gt_5,228). happensAt(walking(mws1gt_5),228).
orientation(mws1gt_1,155,229). coords(mws1gt_1,122,129,229).  visible(mws1gt_1,229). happensAt(walking(mws1gt_1),229). orientation(mws1gt_5,161,229). coords(mws1gt_5,45,126,229).  visible(mws1gt_5,229). happensAt(walking(mws1gt_5),229).
orientation(mws1gt_1,155,230). coords(mws1gt_1,121,128,230).  visible(mws1gt_1,230). happensAt(walking(mws1gt_1),230). orientation(mws1gt_5,161,230). coords(mws1gt_5,45,127,230).  visible(mws1gt_5,230). happensAt(walking(mws1gt_5),230).
orientation(mws1gt_1,155,231). coords(mws1gt_1,120,127,231).  visible(mws1gt_1,231). happensAt(walking(mws1gt_1),231). orientation(mws1gt_5,161,231). coords(mws1gt_5,45,128,231).  visible(mws1gt_5,231). happensAt(walking(mws1gt_5),231).
orientation(mws1gt_1,155,232). coords(mws1gt_1,118,127,232).  visible(mws1gt_1,232). happensAt(walking(mws1gt_1),232). orientation(mws1gt_5,161,232). coords(mws1gt_5,45,128,232).  visible(mws1gt_5,232). happensAt(walking(mws1gt_5),232).
orientation(mws1gt_1,155,233). coords(mws1gt_1,118,126,233).  visible(mws1gt_1,233). happensAt(walking(mws1gt_1),233). orientation(mws1gt_5,161,233). coords(mws1gt_5,45,129,233).  visible(mws1gt_5,233). happensAt(walking(mws1gt_5),233).
orientation(mws1gt_1,155,234). coords(mws1gt_1,117,125,234).  visible(mws1gt_1,234). happensAt(walking(mws1gt_1),234). orientation(mws1gt_5,161,234). coords(mws1gt_5,45,129,234).  visible(mws1gt_5,234). happensAt(walking(mws1gt_5),234).
orientation(mws1gt_1,155,235). coords(mws1gt_1,117,125,235).  visible(mws1gt_1,235). happensAt(walking(mws1gt_1),235). orientation(mws1gt_5,161,235). coords(mws1gt_5,45,130,235).  visible(mws1gt_5,235). happensAt(walking(mws1gt_5),235).
orientation(mws1gt_1,155,236). coords(mws1gt_1,117,125,236).  visible(mws1gt_1,236). happensAt(walking(mws1gt_1),236). orientation(mws1gt_5,161,236). coords(mws1gt_5,44,131,236).  visible(mws1gt_5,236). happensAt(walking(mws1gt_5),236).
orientation(mws1gt_1,155,237). coords(mws1gt_1,117,124,237).  visible(mws1gt_1,237). happensAt(walking(mws1gt_1),237). orientation(mws1gt_5,161,237). coords(mws1gt_5,43,132,237).  visible(mws1gt_5,237). happensAt(walking(mws1gt_5),237).
orientation(mws1gt_1,155,238). coords(mws1gt_1,116,123,238).  visible(mws1gt_1,238). happensAt(walking(mws1gt_1),238). orientation(mws1gt_5,161,238). coords(mws1gt_5,43,133,238).  visible(mws1gt_5,238). happensAt(walking(mws1gt_5),238).
#show holdsAt/2.