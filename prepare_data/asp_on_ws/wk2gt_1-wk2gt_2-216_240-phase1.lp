
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

	orientation(wk2gt_1,129,216). coords(wk2gt_1,115,53,216).  visible(wk2gt_1,216). happensAt(inactive(wk2gt_1),216). orientation(wk2gt_2,141,216). coords(wk2gt_2,98,62,216).  visible(wk2gt_2,216). happensAt(inactive(wk2gt_2),216).
orientation(wk2gt_1,129,217). coords(wk2gt_1,115,53,217).  visible(wk2gt_1,217). happensAt(inactive(wk2gt_1),217). orientation(wk2gt_2,141,217). coords(wk2gt_2,98,62,217).  visible(wk2gt_2,217). happensAt(inactive(wk2gt_2),217).
orientation(wk2gt_1,129,218). coords(wk2gt_1,115,53,218).  visible(wk2gt_1,218). happensAt(inactive(wk2gt_1),218). orientation(wk2gt_2,141,218). coords(wk2gt_2,98,62,218).  visible(wk2gt_2,218). happensAt(inactive(wk2gt_2),218).
orientation(wk2gt_1,129,219). coords(wk2gt_1,115,53,219).  visible(wk2gt_1,219). happensAt(inactive(wk2gt_1),219). orientation(wk2gt_2,141,219). coords(wk2gt_2,98,62,219).  visible(wk2gt_2,219). happensAt(inactive(wk2gt_2),219).
orientation(wk2gt_1,129,220). coords(wk2gt_1,115,53,220).  visible(wk2gt_1,220). happensAt(inactive(wk2gt_1),220). orientation(wk2gt_2,141,220). coords(wk2gt_2,98,62,220).  visible(wk2gt_2,220). happensAt(inactive(wk2gt_2),220).
orientation(wk2gt_1,129,221). coords(wk2gt_1,115,53,221).  visible(wk2gt_1,221). happensAt(inactive(wk2gt_1),221). orientation(wk2gt_2,141,221). coords(wk2gt_2,98,62,221).  visible(wk2gt_2,221). happensAt(inactive(wk2gt_2),221).
orientation(wk2gt_1,126,222). coords(wk2gt_1,115,53,222).  visible(wk2gt_1,222). happensAt(inactive(wk2gt_1),222). orientation(wk2gt_2,141,222). coords(wk2gt_2,98,62,222).  visible(wk2gt_2,222). happensAt(inactive(wk2gt_2),222).
orientation(wk2gt_1,126,223). coords(wk2gt_1,115,53,223).  visible(wk2gt_1,223). happensAt(inactive(wk2gt_1),223). orientation(wk2gt_2,141,223). coords(wk2gt_2,98,62,223).  visible(wk2gt_2,223). happensAt(inactive(wk2gt_2),223).
orientation(wk2gt_1,126,224). coords(wk2gt_1,115,53,224).  visible(wk2gt_1,224). happensAt(inactive(wk2gt_1),224). orientation(wk2gt_2,141,224). coords(wk2gt_2,98,62,224).  visible(wk2gt_2,224). happensAt(inactive(wk2gt_2),224).
orientation(wk2gt_1,126,225). coords(wk2gt_1,115,53,225).  visible(wk2gt_1,225). happensAt(inactive(wk2gt_1),225). orientation(wk2gt_2,141,225). coords(wk2gt_2,98,62,225).  visible(wk2gt_2,225). happensAt(inactive(wk2gt_2),225).
orientation(wk2gt_1,126,226). coords(wk2gt_1,115,53,226).  visible(wk2gt_1,226). happensAt(inactive(wk2gt_1),226). orientation(wk2gt_2,141,226). coords(wk2gt_2,98,62,226).  visible(wk2gt_2,226). happensAt(inactive(wk2gt_2),226).
orientation(wk2gt_1,126,227). coords(wk2gt_1,115,53,227).  visible(wk2gt_1,227). happensAt(inactive(wk2gt_1),227). orientation(wk2gt_2,141,227). coords(wk2gt_2,98,62,227).  visible(wk2gt_2,227). happensAt(inactive(wk2gt_2),227).
orientation(wk2gt_1,126,228). coords(wk2gt_1,115,53,228).  visible(wk2gt_1,228). happensAt(inactive(wk2gt_1),228). orientation(wk2gt_2,141,228). coords(wk2gt_2,98,62,228).  visible(wk2gt_2,228). happensAt(inactive(wk2gt_2),228).
orientation(wk2gt_1,126,229). coords(wk2gt_1,115,53,229).  visible(wk2gt_1,229). happensAt(inactive(wk2gt_1),229). orientation(wk2gt_2,141,229). coords(wk2gt_2,98,62,229).  visible(wk2gt_2,229). happensAt(inactive(wk2gt_2),229).
orientation(wk2gt_1,126,230). coords(wk2gt_1,115,53,230).  visible(wk2gt_1,230). happensAt(inactive(wk2gt_1),230). orientation(wk2gt_2,141,230). coords(wk2gt_2,98,62,230).  visible(wk2gt_2,230). happensAt(inactive(wk2gt_2),230).
orientation(wk2gt_1,126,231). coords(wk2gt_1,115,53,231).  visible(wk2gt_1,231). happensAt(inactive(wk2gt_1),231). orientation(wk2gt_2,141,231). coords(wk2gt_2,98,62,231).  visible(wk2gt_2,231). happensAt(inactive(wk2gt_2),231).
orientation(wk2gt_1,126,232). coords(wk2gt_1,115,53,232).  visible(wk2gt_1,232). happensAt(inactive(wk2gt_1),232). orientation(wk2gt_2,141,232). coords(wk2gt_2,98,62,232).  visible(wk2gt_2,232). happensAt(inactive(wk2gt_2),232).
orientation(wk2gt_1,126,233). coords(wk2gt_1,115,53,233).  visible(wk2gt_1,233). happensAt(inactive(wk2gt_1),233). orientation(wk2gt_2,141,233). coords(wk2gt_2,98,62,233).  visible(wk2gt_2,233). happensAt(inactive(wk2gt_2),233).
orientation(wk2gt_1,126,234). coords(wk2gt_1,115,53,234).  visible(wk2gt_1,234). happensAt(inactive(wk2gt_1),234). orientation(wk2gt_2,141,234). coords(wk2gt_2,98,62,234).  visible(wk2gt_2,234). happensAt(inactive(wk2gt_2),234).
orientation(wk2gt_1,126,235). coords(wk2gt_1,115,53,235).  visible(wk2gt_1,235). happensAt(inactive(wk2gt_1),235). orientation(wk2gt_2,141,235). coords(wk2gt_2,98,62,235).  visible(wk2gt_2,235). happensAt(inactive(wk2gt_2),235).
orientation(wk2gt_1,126,236). coords(wk2gt_1,115,53,236).  visible(wk2gt_1,236). happensAt(inactive(wk2gt_1),236). orientation(wk2gt_2,141,236). coords(wk2gt_2,98,62,236).  visible(wk2gt_2,236). happensAt(inactive(wk2gt_2),236).
orientation(wk2gt_1,126,237). coords(wk2gt_1,115,53,237).  visible(wk2gt_1,237). happensAt(inactive(wk2gt_1),237). orientation(wk2gt_2,141,237). coords(wk2gt_2,98,62,237).  visible(wk2gt_2,237). happensAt(inactive(wk2gt_2),237).
orientation(wk2gt_1,126,238). coords(wk2gt_1,115,53,238).  visible(wk2gt_1,238). happensAt(inactive(wk2gt_1),238). orientation(wk2gt_2,141,238). coords(wk2gt_2,98,62,238).  visible(wk2gt_2,238). happensAt(inactive(wk2gt_2),238).
orientation(wk2gt_1,126,239). coords(wk2gt_1,115,53,239).  visible(wk2gt_1,239). happensAt(inactive(wk2gt_1),239). orientation(wk2gt_2,141,239). coords(wk2gt_2,98,62,239).  visible(wk2gt_2,239). happensAt(inactive(wk2gt_2),239).
orientation(wk2gt_1,126,240). coords(wk2gt_1,115,53,240).  visible(wk2gt_1,240). happensAt(inactive(wk2gt_1),240). orientation(wk2gt_2,141,240). coords(wk2gt_2,98,62,240).  visible(wk2gt_2,240). happensAt(inactive(wk2gt_2),240).
#show holdsAt/2.