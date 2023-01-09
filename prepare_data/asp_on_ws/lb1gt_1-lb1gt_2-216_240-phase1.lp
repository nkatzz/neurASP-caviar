
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

	orientation(lb1gt_1,151,216). coords(lb1gt_1,129,159,216).  visible(lb1gt_1,216). happensAt(walking(lb1gt_1),216). orientation(lb1gt_2,147,216). coords(lb1gt_2,136,120,216).  visible(lb1gt_2,216). happensAt(walking(lb1gt_2),216).
orientation(lb1gt_1,151,217). coords(lb1gt_1,130,159,217).  visible(lb1gt_1,217). happensAt(walking(lb1gt_1),217). orientation(lb1gt_2,147,217). coords(lb1gt_2,137,121,217).  visible(lb1gt_2,217). happensAt(walking(lb1gt_2),217).
orientation(lb1gt_1,151,218). coords(lb1gt_1,130,159,218).  visible(lb1gt_1,218). happensAt(walking(lb1gt_1),218). orientation(lb1gt_2,147,218). coords(lb1gt_2,137,121,218).  visible(lb1gt_2,218). happensAt(walking(lb1gt_2),218).
orientation(lb1gt_1,151,219). coords(lb1gt_1,131,159,219).  visible(lb1gt_1,219). happensAt(walking(lb1gt_1),219). orientation(lb1gt_2,147,219). coords(lb1gt_2,138,121,219).  visible(lb1gt_2,219). happensAt(walking(lb1gt_2),219).
orientation(lb1gt_1,151,220). coords(lb1gt_1,132,159,220).  visible(lb1gt_1,220). happensAt(walking(lb1gt_1),220). orientation(lb1gt_2,147,220). coords(lb1gt_2,138,122,220).  visible(lb1gt_2,220). happensAt(walking(lb1gt_2),220).
orientation(lb1gt_1,151,221). coords(lb1gt_1,133,159,221).  visible(lb1gt_1,221). happensAt(walking(lb1gt_1),221). orientation(lb1gt_2,147,221). coords(lb1gt_2,138,122,221).  visible(lb1gt_2,221). happensAt(walking(lb1gt_2),221).
orientation(lb1gt_1,151,222). coords(lb1gt_1,133,159,222).  visible(lb1gt_1,222). happensAt(walking(lb1gt_1),222). orientation(lb1gt_2,147,222). coords(lb1gt_2,138,122,222).  visible(lb1gt_2,222). happensAt(walking(lb1gt_2),222).
orientation(lb1gt_1,151,223). coords(lb1gt_1,134,159,223).  visible(lb1gt_1,223). happensAt(walking(lb1gt_1),223). orientation(lb1gt_2,147,223). coords(lb1gt_2,138,122,223).  visible(lb1gt_2,223). happensAt(walking(lb1gt_2),223).
orientation(lb1gt_1,151,224). coords(lb1gt_1,134,158,224).  visible(lb1gt_1,224). happensAt(walking(lb1gt_1),224). orientation(lb1gt_2,147,224). coords(lb1gt_2,138,122,224).  visible(lb1gt_2,224). happensAt(walking(lb1gt_2),224).
orientation(lb1gt_1,151,225). coords(lb1gt_1,134,158,225).  visible(lb1gt_1,225). happensAt(walking(lb1gt_1),225). orientation(lb1gt_2,147,225). coords(lb1gt_2,138,122,225).  visible(lb1gt_2,225). happensAt(walking(lb1gt_2),225).
orientation(lb1gt_1,151,226). coords(lb1gt_1,135,158,226).  visible(lb1gt_1,226). happensAt(walking(lb1gt_1),226). orientation(lb1gt_2,147,226). coords(lb1gt_2,139,122,226).  visible(lb1gt_2,226). happensAt(walking(lb1gt_2),226).
orientation(lb1gt_1,151,227). coords(lb1gt_1,135,158,227).  visible(lb1gt_1,227). happensAt(walking(lb1gt_1),227). orientation(lb1gt_2,147,227). coords(lb1gt_2,141,123,227).  visible(lb1gt_2,227). happensAt(walking(lb1gt_2),227).
orientation(lb1gt_1,151,228). coords(lb1gt_1,135,158,228).  visible(lb1gt_1,228). happensAt(walking(lb1gt_1),228). orientation(lb1gt_2,147,228). coords(lb1gt_2,142,123,228).  visible(lb1gt_2,228). happensAt(walking(lb1gt_2),228).
orientation(lb1gt_1,151,229). coords(lb1gt_1,136,158,229).  visible(lb1gt_1,229). happensAt(walking(lb1gt_1),229). orientation(lb1gt_2,147,229). coords(lb1gt_2,144,123,229).  visible(lb1gt_2,229). happensAt(walking(lb1gt_2),229).
orientation(lb1gt_1,151,230). coords(lb1gt_1,136,158,230).  visible(lb1gt_1,230). happensAt(walking(lb1gt_1),230). orientation(lb1gt_2,147,230). coords(lb1gt_2,146,123,230).  visible(lb1gt_2,230). happensAt(walking(lb1gt_2),230).
orientation(lb1gt_1,151,231). coords(lb1gt_1,136,159,231).  visible(lb1gt_1,231). happensAt(walking(lb1gt_1),231). orientation(lb1gt_2,147,231). coords(lb1gt_2,147,123,231).  visible(lb1gt_2,231). happensAt(walking(lb1gt_2),231).
orientation(lb1gt_1,151,232). coords(lb1gt_1,137,159,232).  visible(lb1gt_1,232). happensAt(walking(lb1gt_1),232). orientation(lb1gt_2,147,232). coords(lb1gt_2,147,123,232).  visible(lb1gt_2,232). happensAt(walking(lb1gt_2),232).
orientation(lb1gt_1,151,233). coords(lb1gt_1,139,159,233).  visible(lb1gt_1,233). happensAt(walking(lb1gt_1),233). orientation(lb1gt_2,147,233). coords(lb1gt_2,148,123,233).  visible(lb1gt_2,233). happensAt(walking(lb1gt_2),233).
orientation(lb1gt_1,151,234). coords(lb1gt_1,141,159,234).  visible(lb1gt_1,234). happensAt(walking(lb1gt_1),234). orientation(lb1gt_2,147,234). coords(lb1gt_2,150,123,234).  visible(lb1gt_2,234). happensAt(walking(lb1gt_2),234).
orientation(lb1gt_1,151,235). coords(lb1gt_1,142,159,235).  visible(lb1gt_1,235). happensAt(walking(lb1gt_1),235). orientation(lb1gt_2,149,235). coords(lb1gt_2,150,123,235).  visible(lb1gt_2,235). happensAt(walking(lb1gt_2),235).
orientation(lb1gt_1,151,236). coords(lb1gt_1,144,159,236).  visible(lb1gt_1,236). happensAt(walking(lb1gt_1),236). orientation(lb1gt_2,149,236). coords(lb1gt_2,151,123,236).  visible(lb1gt_2,236). happensAt(walking(lb1gt_2),236).
orientation(lb1gt_1,151,237). coords(lb1gt_1,145,159,237).  visible(lb1gt_1,237). happensAt(walking(lb1gt_1),237). orientation(lb1gt_2,149,237). coords(lb1gt_2,151,123,237).  visible(lb1gt_2,237). happensAt(walking(lb1gt_2),237).
orientation(lb1gt_1,151,238). coords(lb1gt_1,145,159,238).  visible(lb1gt_1,238). happensAt(walking(lb1gt_1),238). orientation(lb1gt_2,149,238). coords(lb1gt_2,152,123,238).  visible(lb1gt_2,238). happensAt(walking(lb1gt_2),238).
orientation(lb1gt_1,151,239). coords(lb1gt_1,146,160,239).  visible(lb1gt_1,239). happensAt(walking(lb1gt_1),239). orientation(lb1gt_2,149,239). coords(lb1gt_2,152,123,239).  visible(lb1gt_2,239). happensAt(walking(lb1gt_2),239).
orientation(lb1gt_1,151,240). coords(lb1gt_1,146,160,240).  visible(lb1gt_1,240). happensAt(walking(lb1gt_1),240). orientation(lb1gt_2,149,240). coords(lb1gt_2,152,124,240).  visible(lb1gt_2,240). happensAt(walking(lb1gt_2),240).
#show holdsAt/2.