
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

	orientation(mws1gt_1,155,216). coords(mws1gt_1,128,132,216).  visible(mws1gt_1,216). happensAt(walking(mws1gt_1),216). orientation(mws1gt_6,146,216). coords(mws1gt_6,55,96,216).  visible(mws1gt_6,216). happensAt(walking(mws1gt_6),216).
orientation(mws1gt_1,155,217). coords(mws1gt_1,127,132,217).  visible(mws1gt_1,217). happensAt(walking(mws1gt_1),217). orientation(mws1gt_6,146,217). coords(mws1gt_6,55,96,217).  visible(mws1gt_6,217). happensAt(walking(mws1gt_6),217).
orientation(mws1gt_1,155,218). coords(mws1gt_1,126,132,218).  visible(mws1gt_1,218). happensAt(walking(mws1gt_1),218). orientation(mws1gt_6,146,218). coords(mws1gt_6,55,97,218).  visible(mws1gt_6,218). happensAt(walking(mws1gt_6),218).
orientation(mws1gt_1,155,219). coords(mws1gt_1,126,132,219).  visible(mws1gt_1,219). happensAt(walking(mws1gt_1),219). orientation(mws1gt_6,146,219). coords(mws1gt_6,55,97,219).  visible(mws1gt_6,219). happensAt(walking(mws1gt_6),219).
orientation(mws1gt_1,155,220). coords(mws1gt_1,126,132,220).  visible(mws1gt_1,220). happensAt(walking(mws1gt_1),220). orientation(mws1gt_6,146,220). coords(mws1gt_6,55,97,220).  visible(mws1gt_6,220). happensAt(walking(mws1gt_6),220).
orientation(mws1gt_1,155,221). coords(mws1gt_1,126,131,221).  visible(mws1gt_1,221). happensAt(walking(mws1gt_1),221). orientation(mws1gt_6,146,221). coords(mws1gt_6,55,98,221).  visible(mws1gt_6,221). happensAt(walking(mws1gt_6),221).
orientation(mws1gt_1,155,222). coords(mws1gt_1,125,131,222).  visible(mws1gt_1,222). happensAt(walking(mws1gt_1),222). orientation(mws1gt_6,146,222). coords(mws1gt_6,55,98,222).  visible(mws1gt_6,222). happensAt(walking(mws1gt_6),222).
orientation(mws1gt_1,155,223). coords(mws1gt_1,125,130,223).  visible(mws1gt_1,223). happensAt(walking(mws1gt_1),223). orientation(mws1gt_6,146,223). coords(mws1gt_6,55,99,223).  visible(mws1gt_6,223). happensAt(walking(mws1gt_6),223).
orientation(mws1gt_1,155,224). coords(mws1gt_1,125,130,224).  visible(mws1gt_1,224). happensAt(walking(mws1gt_1),224). orientation(mws1gt_6,146,224). coords(mws1gt_6,54,99,224).  visible(mws1gt_6,224). happensAt(walking(mws1gt_6),224).
orientation(mws1gt_1,155,225). coords(mws1gt_1,125,130,225).  visible(mws1gt_1,225). happensAt(walking(mws1gt_1),225). orientation(mws1gt_6,146,225). coords(mws1gt_6,53,100,225).  visible(mws1gt_6,225). happensAt(walking(mws1gt_6),225).
orientation(mws1gt_1,155,226). coords(mws1gt_1,125,130,226).  visible(mws1gt_1,226). happensAt(walking(mws1gt_1),226). orientation(mws1gt_6,146,226). coords(mws1gt_6,53,100,226).  visible(mws1gt_6,226). happensAt(walking(mws1gt_6),226).
orientation(mws1gt_1,155,227). coords(mws1gt_1,124,130,227).  visible(mws1gt_1,227). happensAt(walking(mws1gt_1),227). orientation(mws1gt_6,146,227). coords(mws1gt_6,52,101,227).  visible(mws1gt_6,227). happensAt(walking(mws1gt_6),227).
orientation(mws1gt_1,155,228). coords(mws1gt_1,123,129,228).  visible(mws1gt_1,228). happensAt(walking(mws1gt_1),228). orientation(mws1gt_6,146,228). coords(mws1gt_6,52,101,228).  visible(mws1gt_6,228). happensAt(walking(mws1gt_6),228).
orientation(mws1gt_1,155,229). coords(mws1gt_1,122,129,229).  visible(mws1gt_1,229). happensAt(walking(mws1gt_1),229). orientation(mws1gt_6,146,229). coords(mws1gt_6,52,102,229).  visible(mws1gt_6,229). happensAt(walking(mws1gt_6),229).
orientation(mws1gt_1,155,230). coords(mws1gt_1,121,128,230).  visible(mws1gt_1,230). happensAt(walking(mws1gt_1),230). orientation(mws1gt_6,146,230). coords(mws1gt_6,48,102,230).  visible(mws1gt_6,230). happensAt(walking(mws1gt_6),230).
orientation(mws1gt_1,155,231). coords(mws1gt_1,120,127,231).  visible(mws1gt_1,231). happensAt(walking(mws1gt_1),231). orientation(mws1gt_6,146,231). coords(mws1gt_6,48,103,231).  visible(mws1gt_6,231). happensAt(walking(mws1gt_6),231).
orientation(mws1gt_1,155,232). coords(mws1gt_1,118,127,232).  visible(mws1gt_1,232). happensAt(walking(mws1gt_1),232). orientation(mws1gt_6,146,232). coords(mws1gt_6,48,103,232).  visible(mws1gt_6,232). happensAt(walking(mws1gt_6),232).
orientation(mws1gt_1,155,233). coords(mws1gt_1,118,126,233).  visible(mws1gt_1,233). happensAt(walking(mws1gt_1),233). orientation(mws1gt_6,146,233). coords(mws1gt_6,48,103,233).  visible(mws1gt_6,233). happensAt(walking(mws1gt_6),233).
orientation(mws1gt_1,155,234). coords(mws1gt_1,117,125,234).  visible(mws1gt_1,234). happensAt(walking(mws1gt_1),234). orientation(mws1gt_6,146,234). coords(mws1gt_6,48,105,234).  visible(mws1gt_6,234). happensAt(walking(mws1gt_6),234).
orientation(mws1gt_1,155,235). coords(mws1gt_1,117,125,235).  visible(mws1gt_1,235). happensAt(walking(mws1gt_1),235). orientation(mws1gt_6,146,235). coords(mws1gt_6,48,106,235).  visible(mws1gt_6,235). happensAt(walking(mws1gt_6),235).
orientation(mws1gt_1,155,236). coords(mws1gt_1,117,125,236).  visible(mws1gt_1,236). happensAt(walking(mws1gt_1),236). orientation(mws1gt_6,146,236). coords(mws1gt_6,48,106,236).  visible(mws1gt_6,236). happensAt(walking(mws1gt_6),236).
orientation(mws1gt_1,155,237). coords(mws1gt_1,117,124,237).  visible(mws1gt_1,237). happensAt(walking(mws1gt_1),237). orientation(mws1gt_6,146,237). coords(mws1gt_6,44,107,237).  visible(mws1gt_6,237). happensAt(walking(mws1gt_6),237).
orientation(mws1gt_1,155,238). coords(mws1gt_1,116,123,238).  visible(mws1gt_1,238). happensAt(walking(mws1gt_1),238). orientation(mws1gt_6,146,238). coords(mws1gt_6,44,108,238).  visible(mws1gt_6,238). happensAt(walking(mws1gt_6),238).
orientation(mws1gt_1,155,239). coords(mws1gt_1,116,123,239).  visible(mws1gt_1,239). happensAt(walking(mws1gt_1),239). orientation(mws1gt_6,146,239). coords(mws1gt_6,44,109,239).  visible(mws1gt_6,239). happensAt(walking(mws1gt_6),239).
orientation(mws1gt_1,155,240). coords(mws1gt_1,116,123,240).  visible(mws1gt_1,240). happensAt(walking(mws1gt_1),240). orientation(mws1gt_6,146,240). coords(mws1gt_6,44,110,240).  visible(mws1gt_6,240). happensAt(walking(mws1gt_6),240).
#show holdsAt/2.