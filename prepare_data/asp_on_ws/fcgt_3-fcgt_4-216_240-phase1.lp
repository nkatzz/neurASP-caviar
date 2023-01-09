
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

	orientation(fcgt_3,149,216). coords(fcgt_3,100,168,216).  visible(fcgt_3,216). happensAt(walking(fcgt_3),216). orientation(fcgt_4,172,216). coords(fcgt_4,82,205,216).  visible(fcgt_4,216). happensAt(walking(fcgt_4),216).
orientation(fcgt_3,149,217). coords(fcgt_3,100,169,217).  visible(fcgt_3,217). happensAt(walking(fcgt_3),217). orientation(fcgt_4,172,217). coords(fcgt_4,82,205,217).  visible(fcgt_4,217). happensAt(walking(fcgt_4),217).
orientation(fcgt_3,149,218). coords(fcgt_3,100,169,218).  visible(fcgt_3,218). happensAt(walking(fcgt_3),218). orientation(fcgt_4,172,218). coords(fcgt_4,82,205,218).  visible(fcgt_4,218). happensAt(walking(fcgt_4),218).
orientation(fcgt_3,149,219). coords(fcgt_3,100,169,219).  visible(fcgt_3,219). happensAt(walking(fcgt_3),219). orientation(fcgt_4,172,219). coords(fcgt_4,81,205,219).  visible(fcgt_4,219). happensAt(walking(fcgt_4),219).
orientation(fcgt_3,149,220). coords(fcgt_3,100,170,220).  visible(fcgt_3,220). happensAt(walking(fcgt_3),220). orientation(fcgt_4,172,220). coords(fcgt_4,81,205,220).  visible(fcgt_4,220). happensAt(walking(fcgt_4),220).
orientation(fcgt_3,149,221). coords(fcgt_3,100,170,221).  visible(fcgt_3,221). happensAt(walking(fcgt_3),221). orientation(fcgt_4,172,221). coords(fcgt_4,81,205,221).  visible(fcgt_4,221). happensAt(walking(fcgt_4),221).
orientation(fcgt_3,149,222). coords(fcgt_3,100,170,222).  visible(fcgt_3,222). happensAt(walking(fcgt_3),222). orientation(fcgt_4,172,222). coords(fcgt_4,81,205,222).  visible(fcgt_4,222). happensAt(walking(fcgt_4),222).
orientation(fcgt_3,149,223). coords(fcgt_3,100,170,223).  visible(fcgt_3,223). happensAt(walking(fcgt_3),223). orientation(fcgt_4,166,223). coords(fcgt_4,81,205,223).  visible(fcgt_4,223). happensAt(inactive(fcgt_4),223).
orientation(fcgt_3,149,224). coords(fcgt_3,100,170,224).  visible(fcgt_3,224). happensAt(walking(fcgt_3),224). orientation(fcgt_4,166,224). coords(fcgt_4,81,205,224).  visible(fcgt_4,224). happensAt(inactive(fcgt_4),224).
orientation(fcgt_3,149,225). coords(fcgt_3,100,171,225).  visible(fcgt_3,225). happensAt(walking(fcgt_3),225). orientation(fcgt_4,166,225). coords(fcgt_4,80,205,225).  visible(fcgt_4,225). happensAt(inactive(fcgt_4),225).
orientation(fcgt_3,149,226). coords(fcgt_3,100,171,226).  visible(fcgt_3,226). happensAt(walking(fcgt_3),226). orientation(fcgt_4,166,226). coords(fcgt_4,80,205,226).  visible(fcgt_4,226). happensAt(inactive(fcgt_4),226).
orientation(fcgt_3,157,227). coords(fcgt_3,100,171,227).  visible(fcgt_3,227). happensAt(walking(fcgt_3),227). orientation(fcgt_4,166,227). coords(fcgt_4,80,205,227).  visible(fcgt_4,227). happensAt(inactive(fcgt_4),227).
orientation(fcgt_3,159,228). coords(fcgt_3,100,171,228).  visible(fcgt_3,228). happensAt(walking(fcgt_3),228). orientation(fcgt_4,166,228). coords(fcgt_4,80,205,228).  visible(fcgt_4,228). happensAt(inactive(fcgt_4),228).
orientation(fcgt_3,159,229). coords(fcgt_3,100,172,229).  visible(fcgt_3,229). happensAt(walking(fcgt_3),229). orientation(fcgt_4,166,229). coords(fcgt_4,80,205,229).  visible(fcgt_4,229). happensAt(inactive(fcgt_4),229).
orientation(fcgt_3,159,230). coords(fcgt_3,100,172,230).  visible(fcgt_3,230). happensAt(walking(fcgt_3),230). orientation(fcgt_4,166,230). coords(fcgt_4,80,205,230).  visible(fcgt_4,230). happensAt(inactive(fcgt_4),230).
orientation(fcgt_3,159,231). coords(fcgt_3,100,172,231).  visible(fcgt_3,231). happensAt(walking(fcgt_3),231). orientation(fcgt_4,166,231). coords(fcgt_4,80,205,231).  visible(fcgt_4,231). happensAt(inactive(fcgt_4),231).
orientation(fcgt_3,159,232). coords(fcgt_3,100,172,232).  visible(fcgt_3,232). happensAt(walking(fcgt_3),232). orientation(fcgt_4,166,232). coords(fcgt_4,80,205,232).  visible(fcgt_4,232). happensAt(inactive(fcgt_4),232).
orientation(fcgt_3,159,233). coords(fcgt_3,100,172,233).  visible(fcgt_3,233). happensAt(walking(fcgt_3),233). orientation(fcgt_4,166,233). coords(fcgt_4,80,205,233).  visible(fcgt_4,233). happensAt(inactive(fcgt_4),233).
orientation(fcgt_3,159,234). coords(fcgt_3,100,172,234).  visible(fcgt_3,234). happensAt(walking(fcgt_3),234). orientation(fcgt_4,166,234). coords(fcgt_4,80,205,234).  visible(fcgt_4,234). happensAt(inactive(fcgt_4),234).
orientation(fcgt_3,159,235). coords(fcgt_3,100,173,235).  visible(fcgt_3,235). happensAt(walking(fcgt_3),235). orientation(fcgt_4,166,235). coords(fcgt_4,80,205,235).  visible(fcgt_4,235). happensAt(inactive(fcgt_4),235).
orientation(fcgt_3,159,236). coords(fcgt_3,100,173,236).  visible(fcgt_3,236). happensAt(walking(fcgt_3),236). orientation(fcgt_4,166,236). coords(fcgt_4,80,205,236).  visible(fcgt_4,236). happensAt(inactive(fcgt_4),236).
orientation(fcgt_3,159,237). coords(fcgt_3,100,173,237).  visible(fcgt_3,237). happensAt(walking(fcgt_3),237). orientation(fcgt_4,166,237). coords(fcgt_4,80,205,237).  visible(fcgt_4,237). happensAt(inactive(fcgt_4),237).
orientation(fcgt_3,159,238). coords(fcgt_3,100,174,238).  visible(fcgt_3,238). happensAt(walking(fcgt_3),238). orientation(fcgt_4,166,238). coords(fcgt_4,80,205,238).  visible(fcgt_4,238). happensAt(inactive(fcgt_4),238).
orientation(fcgt_3,163,239). coords(fcgt_3,101,174,239).  visible(fcgt_3,239). happensAt(walking(fcgt_3),239). orientation(fcgt_4,166,239). coords(fcgt_4,80,205,239).  visible(fcgt_4,239). happensAt(inactive(fcgt_4),239).
orientation(fcgt_3,166,240). coords(fcgt_3,101,175,240).  visible(fcgt_3,240). happensAt(walking(fcgt_3),240). orientation(fcgt_4,166,240). coords(fcgt_4,80,205,240).  visible(fcgt_4,240). happensAt(inactive(fcgt_4),240).
#show holdsAt/2.