
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

	orientation(mc1gt_2,152,216). coords(mc1gt_2,115,125,216).  visible(mc1gt_2,216). happensAt(walking(mc1gt_2),216). orientation(mc1gt_3,146,216). coords(mc1gt_3,145,84,216).  visible(mc1gt_3,216). happensAt(walking(mc1gt_3),216).
orientation(mc1gt_2,152,217). coords(mc1gt_2,116,125,217).  visible(mc1gt_2,217). happensAt(walking(mc1gt_2),217). orientation(mc1gt_3,146,217). coords(mc1gt_3,148,85,217).  visible(mc1gt_3,217). happensAt(walking(mc1gt_3),217).
orientation(mc1gt_2,152,218). coords(mc1gt_2,117,125,218).  visible(mc1gt_2,218). happensAt(walking(mc1gt_2),218). orientation(mc1gt_3,146,218). coords(mc1gt_3,151,86,218).  visible(mc1gt_3,218). happensAt(walking(mc1gt_3),218).
orientation(mc1gt_2,152,219). coords(mc1gt_2,119,126,219).  visible(mc1gt_2,219). happensAt(walking(mc1gt_2),219). orientation(mc1gt_3,146,219). coords(mc1gt_3,153,87,219).  visible(mc1gt_3,219). happensAt(walking(mc1gt_3),219).
orientation(mc1gt_2,152,220). coords(mc1gt_2,119,126,220).  visible(mc1gt_2,220). happensAt(walking(mc1gt_2),220). orientation(mc1gt_3,146,220). coords(mc1gt_3,154,87,220).  visible(mc1gt_3,220). happensAt(walking(mc1gt_3),220).
orientation(mc1gt_2,152,221). coords(mc1gt_2,121,127,221).  visible(mc1gt_2,221). happensAt(walking(mc1gt_2),221). orientation(mc1gt_3,146,221). coords(mc1gt_3,154,87,221).  visible(mc1gt_3,221). happensAt(walking(mc1gt_3),221).
orientation(mc1gt_2,152,222). coords(mc1gt_2,121,127,222).  visible(mc1gt_2,222). happensAt(walking(mc1gt_2),222). orientation(mc1gt_3,146,222). coords(mc1gt_3,154,87,222).  visible(mc1gt_3,222). happensAt(walking(mc1gt_3),222).
orientation(mc1gt_2,152,223). coords(mc1gt_2,122,127,223).  visible(mc1gt_2,223). happensAt(walking(mc1gt_2),223). orientation(mc1gt_3,146,223). coords(mc1gt_3,155,88,223).  visible(mc1gt_3,223). happensAt(walking(mc1gt_3),223).
orientation(mc1gt_2,149,224). coords(mc1gt_2,122,127,224).  visible(mc1gt_2,224). happensAt(walking(mc1gt_2),224). orientation(mc1gt_3,146,224). coords(mc1gt_3,155,88,224).  visible(mc1gt_3,224). happensAt(walking(mc1gt_3),224).
orientation(mc1gt_2,148,225). coords(mc1gt_2,123,127,225).  visible(mc1gt_2,225). happensAt(walking(mc1gt_2),225). orientation(mc1gt_3,146,225). coords(mc1gt_3,157,88,225).  visible(mc1gt_3,225). happensAt(walking(mc1gt_3),225).
orientation(mc1gt_2,147,226). coords(mc1gt_2,123,127,226).  visible(mc1gt_2,226). happensAt(walking(mc1gt_2),226). orientation(mc1gt_3,146,226). coords(mc1gt_3,160,88,226).  visible(mc1gt_3,226). happensAt(walking(mc1gt_3),226).
orientation(mc1gt_2,147,227). coords(mc1gt_2,123,127,227).  visible(mc1gt_2,227). happensAt(walking(mc1gt_2),227). orientation(mc1gt_3,146,227). coords(mc1gt_3,163,88,227).  visible(mc1gt_3,227). happensAt(walking(mc1gt_3),227).
orientation(mc1gt_2,147,228). coords(mc1gt_2,123,129,228).  visible(mc1gt_2,228). happensAt(walking(mc1gt_2),228). orientation(mc1gt_3,146,228). coords(mc1gt_3,165,88,228).  visible(mc1gt_3,228). happensAt(walking(mc1gt_3),228).
orientation(mc1gt_2,147,229). coords(mc1gt_2,124,130,229).  visible(mc1gt_2,229). happensAt(walking(mc1gt_2),229). orientation(mc1gt_3,146,229). coords(mc1gt_3,167,88,229).  visible(mc1gt_3,229). happensAt(walking(mc1gt_3),229).
orientation(mc1gt_2,147,230). coords(mc1gt_2,125,131,230).  visible(mc1gt_2,230). happensAt(walking(mc1gt_2),230). orientation(mc1gt_3,143,230). coords(mc1gt_3,169,88,230).  visible(mc1gt_3,230). happensAt(walking(mc1gt_3),230).
orientation(mc1gt_2,146,231). coords(mc1gt_2,127,132,231).  visible(mc1gt_2,231). happensAt(walking(mc1gt_2),231). orientation(mc1gt_3,141,231). coords(mc1gt_3,171,88,231).  visible(mc1gt_3,231). happensAt(walking(mc1gt_3),231).
orientation(mc1gt_2,145,232). coords(mc1gt_2,129,133,232).  visible(mc1gt_2,232). happensAt(walking(mc1gt_2),232). orientation(mc1gt_3,139,232). coords(mc1gt_3,172,88,232).  visible(mc1gt_3,232). happensAt(walking(mc1gt_3),232).
orientation(mc1gt_2,147,233). coords(mc1gt_2,130,134,233).  visible(mc1gt_2,233). happensAt(walking(mc1gt_2),233). orientation(mc1gt_3,141,233). coords(mc1gt_3,173,88,233).  visible(mc1gt_3,233). happensAt(walking(mc1gt_3),233).
orientation(mc1gt_2,147,234). coords(mc1gt_2,131,134,234).  visible(mc1gt_2,234). happensAt(walking(mc1gt_2),234). orientation(mc1gt_3,140,234). coords(mc1gt_3,173,88,234).  visible(mc1gt_3,234). happensAt(walking(mc1gt_3),234).
orientation(mc1gt_2,147,235). coords(mc1gt_2,131,135,235).  visible(mc1gt_2,235). happensAt(walking(mc1gt_2),235). orientation(mc1gt_3,141,235). coords(mc1gt_3,174,89,235).  visible(mc1gt_3,235). happensAt(walking(mc1gt_3),235).
orientation(mc1gt_2,146,236). coords(mc1gt_2,131,135,236).  visible(mc1gt_2,236). happensAt(walking(mc1gt_2),236). orientation(mc1gt_3,138,236). coords(mc1gt_3,174,89,236).  visible(mc1gt_3,236). happensAt(walking(mc1gt_3),236).
orientation(mc1gt_2,146,237). coords(mc1gt_2,132,135,237).  visible(mc1gt_2,237). happensAt(walking(mc1gt_2),237). orientation(mc1gt_3,137,237). coords(mc1gt_3,175,90,237).  visible(mc1gt_3,237). happensAt(walking(mc1gt_3),237).
orientation(mc1gt_2,146,238). coords(mc1gt_2,132,136,238).  visible(mc1gt_2,238). happensAt(walking(mc1gt_2),238). orientation(mc1gt_3,137,238). coords(mc1gt_3,176,92,238).  visible(mc1gt_3,238). happensAt(walking(mc1gt_3),238).
orientation(mc1gt_2,146,239). coords(mc1gt_2,132,136,239).  visible(mc1gt_2,239). happensAt(walking(mc1gt_2),239). orientation(mc1gt_3,139,239). coords(mc1gt_3,178,93,239).  visible(mc1gt_3,239). happensAt(walking(mc1gt_3),239).
orientation(mc1gt_2,146,240). coords(mc1gt_2,133,137,240).  visible(mc1gt_2,240). happensAt(walking(mc1gt_2),240). orientation(mc1gt_3,139,240). coords(mc1gt_3,181,94,240).  visible(mc1gt_3,240). happensAt(walking(mc1gt_3),240).
#show holdsAt/2.