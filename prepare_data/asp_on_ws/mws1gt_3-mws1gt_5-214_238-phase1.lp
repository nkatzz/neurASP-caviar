
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

	orientation(mws1gt_3,162,214). coords(mws1gt_3,126,154,214).  visible(mws1gt_3,214). happensAt(walking(mws1gt_3),214). orientation(mws1gt_5,161,214). coords(mws1gt_5,45,113,214).  visible(mws1gt_5,214). happensAt(walking(mws1gt_5),214).
orientation(mws1gt_3,162,215). coords(mws1gt_3,125,154,215).  visible(mws1gt_3,215). happensAt(walking(mws1gt_3),215). orientation(mws1gt_5,161,215). coords(mws1gt_5,46,116,215).  visible(mws1gt_5,215). happensAt(walking(mws1gt_5),215).
orientation(mws1gt_3,162,216). coords(mws1gt_3,125,154,216).  visible(mws1gt_3,216). happensAt(walking(mws1gt_3),216). orientation(mws1gt_5,161,216). coords(mws1gt_5,46,117,216).  visible(mws1gt_5,216). happensAt(walking(mws1gt_5),216).
orientation(mws1gt_3,162,217). coords(mws1gt_3,125,154,217).  visible(mws1gt_3,217). happensAt(walking(mws1gt_3),217). orientation(mws1gt_5,161,217). coords(mws1gt_5,46,117,217).  visible(mws1gt_5,217). happensAt(walking(mws1gt_5),217).
orientation(mws1gt_3,162,218). coords(mws1gt_3,124,153,218).  visible(mws1gt_3,218). happensAt(walking(mws1gt_3),218). orientation(mws1gt_5,161,218). coords(mws1gt_5,46,118,218).  visible(mws1gt_5,218). happensAt(walking(mws1gt_5),218).
orientation(mws1gt_3,162,219). coords(mws1gt_3,124,153,219).  visible(mws1gt_3,219). happensAt(walking(mws1gt_3),219). orientation(mws1gt_5,161,219). coords(mws1gt_5,46,118,219).  visible(mws1gt_5,219). happensAt(walking(mws1gt_5),219).
orientation(mws1gt_3,162,220). coords(mws1gt_3,123,153,220).  visible(mws1gt_3,220). happensAt(walking(mws1gt_3),220). orientation(mws1gt_5,161,220). coords(mws1gt_5,46,119,220).  visible(mws1gt_5,220). happensAt(walking(mws1gt_5),220).
orientation(mws1gt_3,162,221). coords(mws1gt_3,123,153,221).  visible(mws1gt_3,221). happensAt(walking(mws1gt_3),221). orientation(mws1gt_5,161,221). coords(mws1gt_5,46,120,221).  visible(mws1gt_5,221). happensAt(walking(mws1gt_5),221).
orientation(mws1gt_3,162,222). coords(mws1gt_3,123,153,222).  visible(mws1gt_3,222). happensAt(walking(mws1gt_3),222). orientation(mws1gt_5,161,222). coords(mws1gt_5,46,120,222).  visible(mws1gt_5,222). happensAt(walking(mws1gt_5),222).
orientation(mws1gt_3,162,223). coords(mws1gt_3,122,152,223).  visible(mws1gt_3,223). happensAt(walking(mws1gt_3),223). orientation(mws1gt_5,161,223). coords(mws1gt_5,46,120,223).  visible(mws1gt_5,223). happensAt(walking(mws1gt_5),223).
orientation(mws1gt_3,162,224). coords(mws1gt_3,121,152,224).  visible(mws1gt_3,224). happensAt(walking(mws1gt_3),224). orientation(mws1gt_5,161,224). coords(mws1gt_5,46,120,224).  visible(mws1gt_5,224). happensAt(walking(mws1gt_5),224).
orientation(mws1gt_3,162,225). coords(mws1gt_3,120,151,225).  visible(mws1gt_3,225). happensAt(walking(mws1gt_3),225). orientation(mws1gt_5,161,225). coords(mws1gt_5,46,120,225).  visible(mws1gt_5,225). happensAt(walking(mws1gt_5),225).
orientation(mws1gt_3,162,226). coords(mws1gt_3,118,150,226).  visible(mws1gt_3,226). happensAt(walking(mws1gt_3),226). orientation(mws1gt_5,161,226). coords(mws1gt_5,45,120,226).  visible(mws1gt_5,226). happensAt(walking(mws1gt_5),226).
orientation(mws1gt_3,162,227). coords(mws1gt_3,116,149,227).  visible(mws1gt_3,227). happensAt(walking(mws1gt_3),227). orientation(mws1gt_5,161,227). coords(mws1gt_5,45,121,227).  visible(mws1gt_5,227). happensAt(walking(mws1gt_5),227).
orientation(mws1gt_3,162,228). coords(mws1gt_3,115,149,228).  visible(mws1gt_3,228). happensAt(walking(mws1gt_3),228). orientation(mws1gt_5,161,228). coords(mws1gt_5,45,124,228).  visible(mws1gt_5,228). happensAt(walking(mws1gt_5),228).
orientation(mws1gt_3,162,229). coords(mws1gt_3,113,149,229).  visible(mws1gt_3,229). happensAt(walking(mws1gt_3),229). orientation(mws1gt_5,161,229). coords(mws1gt_5,45,126,229).  visible(mws1gt_5,229). happensAt(walking(mws1gt_5),229).
orientation(mws1gt_3,162,230). coords(mws1gt_3,113,149,230).  visible(mws1gt_3,230). happensAt(walking(mws1gt_3),230). orientation(mws1gt_5,161,230). coords(mws1gt_5,45,127,230).  visible(mws1gt_5,230). happensAt(walking(mws1gt_5),230).
orientation(mws1gt_3,162,231). coords(mws1gt_3,113,149,231).  visible(mws1gt_3,231). happensAt(walking(mws1gt_3),231). orientation(mws1gt_5,161,231). coords(mws1gt_5,45,128,231).  visible(mws1gt_5,231). happensAt(walking(mws1gt_5),231).
orientation(mws1gt_3,162,232). coords(mws1gt_3,113,149,232).  visible(mws1gt_3,232). happensAt(walking(mws1gt_3),232). orientation(mws1gt_5,161,232). coords(mws1gt_5,45,128,232).  visible(mws1gt_5,232). happensAt(walking(mws1gt_5),232).
orientation(mws1gt_3,162,233). coords(mws1gt_3,112,148,233).  visible(mws1gt_3,233). happensAt(walking(mws1gt_3),233). orientation(mws1gt_5,161,233). coords(mws1gt_5,45,129,233).  visible(mws1gt_5,233). happensAt(walking(mws1gt_5),233).
orientation(mws1gt_3,162,234). coords(mws1gt_3,112,148,234).  visible(mws1gt_3,234). happensAt(walking(mws1gt_3),234). orientation(mws1gt_5,161,234). coords(mws1gt_5,45,129,234).  visible(mws1gt_5,234). happensAt(walking(mws1gt_5),234).
orientation(mws1gt_3,162,235). coords(mws1gt_3,112,148,235).  visible(mws1gt_3,235). happensAt(walking(mws1gt_3),235). orientation(mws1gt_5,161,235). coords(mws1gt_5,45,130,235).  visible(mws1gt_5,235). happensAt(walking(mws1gt_5),235).
orientation(mws1gt_3,157,236). coords(mws1gt_3,111,148,236).  visible(mws1gt_3,236). happensAt(walking(mws1gt_3),236). orientation(mws1gt_5,161,236). coords(mws1gt_5,44,131,236).  visible(mws1gt_5,236). happensAt(walking(mws1gt_5),236).
orientation(mws1gt_3,157,237). coords(mws1gt_3,110,147,237).  visible(mws1gt_3,237). happensAt(walking(mws1gt_3),237). orientation(mws1gt_5,161,237). coords(mws1gt_5,43,132,237).  visible(mws1gt_5,237). happensAt(walking(mws1gt_5),237).
orientation(mws1gt_3,157,238). coords(mws1gt_3,109,147,238).  visible(mws1gt_3,238). happensAt(walking(mws1gt_3),238). orientation(mws1gt_5,161,238). coords(mws1gt_5,43,133,238).  visible(mws1gt_5,238). happensAt(walking(mws1gt_5),238).
#show holdsAt/2.