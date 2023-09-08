
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

	orientation(mws1gt_3,162,216). coords(mws1gt_3,125,154,216).  visible(mws1gt_3,216). happensAt(walking(mws1gt_3),216). orientation(mws1gt_6,146,216). coords(mws1gt_6,55,96,216).  visible(mws1gt_6,216). happensAt(walking(mws1gt_6),216).
orientation(mws1gt_3,162,217). coords(mws1gt_3,125,154,217).  visible(mws1gt_3,217). happensAt(walking(mws1gt_3),217). orientation(mws1gt_6,146,217). coords(mws1gt_6,55,96,217).  visible(mws1gt_6,217). happensAt(walking(mws1gt_6),217).
orientation(mws1gt_3,162,218). coords(mws1gt_3,124,153,218).  visible(mws1gt_3,218). happensAt(walking(mws1gt_3),218). orientation(mws1gt_6,146,218). coords(mws1gt_6,55,97,218).  visible(mws1gt_6,218). happensAt(walking(mws1gt_6),218).
orientation(mws1gt_3,162,219). coords(mws1gt_3,124,153,219).  visible(mws1gt_3,219). happensAt(walking(mws1gt_3),219). orientation(mws1gt_6,146,219). coords(mws1gt_6,55,97,219).  visible(mws1gt_6,219). happensAt(walking(mws1gt_6),219).
orientation(mws1gt_3,162,220). coords(mws1gt_3,123,153,220).  visible(mws1gt_3,220). happensAt(walking(mws1gt_3),220). orientation(mws1gt_6,146,220). coords(mws1gt_6,55,97,220).  visible(mws1gt_6,220). happensAt(walking(mws1gt_6),220).
orientation(mws1gt_3,162,221). coords(mws1gt_3,123,153,221).  visible(mws1gt_3,221). happensAt(walking(mws1gt_3),221). orientation(mws1gt_6,146,221). coords(mws1gt_6,55,98,221).  visible(mws1gt_6,221). happensAt(walking(mws1gt_6),221).
orientation(mws1gt_3,162,222). coords(mws1gt_3,123,153,222).  visible(mws1gt_3,222). happensAt(walking(mws1gt_3),222). orientation(mws1gt_6,146,222). coords(mws1gt_6,55,98,222).  visible(mws1gt_6,222). happensAt(walking(mws1gt_6),222).
orientation(mws1gt_3,162,223). coords(mws1gt_3,122,152,223).  visible(mws1gt_3,223). happensAt(walking(mws1gt_3),223). orientation(mws1gt_6,146,223). coords(mws1gt_6,55,99,223).  visible(mws1gt_6,223). happensAt(walking(mws1gt_6),223).
orientation(mws1gt_3,162,224). coords(mws1gt_3,121,152,224).  visible(mws1gt_3,224). happensAt(walking(mws1gt_3),224). orientation(mws1gt_6,146,224). coords(mws1gt_6,54,99,224).  visible(mws1gt_6,224). happensAt(walking(mws1gt_6),224).
orientation(mws1gt_3,162,225). coords(mws1gt_3,120,151,225).  visible(mws1gt_3,225). happensAt(walking(mws1gt_3),225). orientation(mws1gt_6,146,225). coords(mws1gt_6,53,100,225).  visible(mws1gt_6,225). happensAt(walking(mws1gt_6),225).
orientation(mws1gt_3,162,226). coords(mws1gt_3,118,150,226).  visible(mws1gt_3,226). happensAt(walking(mws1gt_3),226). orientation(mws1gt_6,146,226). coords(mws1gt_6,53,100,226).  visible(mws1gt_6,226). happensAt(walking(mws1gt_6),226).
orientation(mws1gt_3,162,227). coords(mws1gt_3,116,149,227).  visible(mws1gt_3,227). happensAt(walking(mws1gt_3),227). orientation(mws1gt_6,146,227). coords(mws1gt_6,52,101,227).  visible(mws1gt_6,227). happensAt(walking(mws1gt_6),227).
orientation(mws1gt_3,162,228). coords(mws1gt_3,115,149,228).  visible(mws1gt_3,228). happensAt(walking(mws1gt_3),228). orientation(mws1gt_6,146,228). coords(mws1gt_6,52,101,228).  visible(mws1gt_6,228). happensAt(walking(mws1gt_6),228).
orientation(mws1gt_3,162,229). coords(mws1gt_3,113,149,229).  visible(mws1gt_3,229). happensAt(walking(mws1gt_3),229). orientation(mws1gt_6,146,229). coords(mws1gt_6,52,102,229).  visible(mws1gt_6,229). happensAt(walking(mws1gt_6),229).
orientation(mws1gt_3,162,230). coords(mws1gt_3,113,149,230).  visible(mws1gt_3,230). happensAt(walking(mws1gt_3),230). orientation(mws1gt_6,146,230). coords(mws1gt_6,48,102,230).  visible(mws1gt_6,230). happensAt(walking(mws1gt_6),230).
orientation(mws1gt_3,162,231). coords(mws1gt_3,113,149,231).  visible(mws1gt_3,231). happensAt(walking(mws1gt_3),231). orientation(mws1gt_6,146,231). coords(mws1gt_6,48,103,231).  visible(mws1gt_6,231). happensAt(walking(mws1gt_6),231).
orientation(mws1gt_3,162,232). coords(mws1gt_3,113,149,232).  visible(mws1gt_3,232). happensAt(walking(mws1gt_3),232). orientation(mws1gt_6,146,232). coords(mws1gt_6,48,103,232).  visible(mws1gt_6,232). happensAt(walking(mws1gt_6),232).
orientation(mws1gt_3,162,233). coords(mws1gt_3,112,148,233).  visible(mws1gt_3,233). happensAt(walking(mws1gt_3),233). orientation(mws1gt_6,146,233). coords(mws1gt_6,48,103,233).  visible(mws1gt_6,233). happensAt(walking(mws1gt_6),233).
orientation(mws1gt_3,162,234). coords(mws1gt_3,112,148,234).  visible(mws1gt_3,234). happensAt(walking(mws1gt_3),234). orientation(mws1gt_6,146,234). coords(mws1gt_6,48,105,234).  visible(mws1gt_6,234). happensAt(walking(mws1gt_6),234).
orientation(mws1gt_3,162,235). coords(mws1gt_3,112,148,235).  visible(mws1gt_3,235). happensAt(walking(mws1gt_3),235). orientation(mws1gt_6,146,235). coords(mws1gt_6,48,106,235).  visible(mws1gt_6,235). happensAt(walking(mws1gt_6),235).
orientation(mws1gt_3,157,236). coords(mws1gt_3,111,148,236).  visible(mws1gt_3,236). happensAt(walking(mws1gt_3),236). orientation(mws1gt_6,146,236). coords(mws1gt_6,48,106,236).  visible(mws1gt_6,236). happensAt(walking(mws1gt_6),236).
orientation(mws1gt_3,157,237). coords(mws1gt_3,110,147,237).  visible(mws1gt_3,237). happensAt(walking(mws1gt_3),237). orientation(mws1gt_6,146,237). coords(mws1gt_6,44,107,237).  visible(mws1gt_6,237). happensAt(walking(mws1gt_6),237).
orientation(mws1gt_3,157,238). coords(mws1gt_3,109,147,238).  visible(mws1gt_3,238). happensAt(walking(mws1gt_3),238). orientation(mws1gt_6,146,238). coords(mws1gt_6,44,108,238).  visible(mws1gt_6,238). happensAt(walking(mws1gt_6),238).
orientation(mws1gt_3,157,239). coords(mws1gt_3,107,147,239).  visible(mws1gt_3,239). happensAt(walking(mws1gt_3),239). orientation(mws1gt_6,146,239). coords(mws1gt_6,44,109,239).  visible(mws1gt_6,239). happensAt(walking(mws1gt_6),239).
orientation(mws1gt_3,157,240). coords(mws1gt_3,106,146,240).  visible(mws1gt_3,240). happensAt(walking(mws1gt_3),240). orientation(mws1gt_6,146,240). coords(mws1gt_6,44,110,240).  visible(mws1gt_6,240). happensAt(walking(mws1gt_6),240).
#show holdsAt/2.