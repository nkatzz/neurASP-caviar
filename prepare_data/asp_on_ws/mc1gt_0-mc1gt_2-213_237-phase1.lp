
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

	orientation(mc1gt_0,147,213). coords(mc1gt_0,144,122,213).  visible(mc1gt_0,213). happensAt(walking(mc1gt_0),213). orientation(mc1gt_2,148,213). coords(mc1gt_2,111,125,213).  visible(mc1gt_2,213). happensAt(walking(mc1gt_2),213).
orientation(mc1gt_0,147,214). coords(mc1gt_0,144,123,214).  visible(mc1gt_0,214). happensAt(walking(mc1gt_0),214). orientation(mc1gt_2,148,214). coords(mc1gt_2,111,125,214).  visible(mc1gt_2,214). happensAt(walking(mc1gt_2),214).
orientation(mc1gt_0,147,215). coords(mc1gt_0,146,125,215).  visible(mc1gt_0,215). happensAt(walking(mc1gt_0),215). orientation(mc1gt_2,152,215). coords(mc1gt_2,113,125,215).  visible(mc1gt_2,215). happensAt(walking(mc1gt_2),215).
orientation(mc1gt_0,147,216). coords(mc1gt_0,147,126,216).  visible(mc1gt_0,216). happensAt(walking(mc1gt_0),216). orientation(mc1gt_2,152,216). coords(mc1gt_2,115,125,216).  visible(mc1gt_2,216). happensAt(walking(mc1gt_2),216).
orientation(mc1gt_0,147,217). coords(mc1gt_0,149,126,217).  visible(mc1gt_0,217). happensAt(walking(mc1gt_0),217). orientation(mc1gt_2,152,217). coords(mc1gt_2,116,125,217).  visible(mc1gt_2,217). happensAt(walking(mc1gt_2),217).
orientation(mc1gt_0,147,218). coords(mc1gt_0,150,127,218).  visible(mc1gt_0,218). happensAt(walking(mc1gt_0),218). orientation(mc1gt_2,152,218). coords(mc1gt_2,117,125,218).  visible(mc1gt_2,218). happensAt(walking(mc1gt_2),218).
orientation(mc1gt_0,147,219). coords(mc1gt_0,151,128,219).  visible(mc1gt_0,219). happensAt(walking(mc1gt_0),219). orientation(mc1gt_2,152,219). coords(mc1gt_2,119,126,219).  visible(mc1gt_2,219). happensAt(walking(mc1gt_2),219).
orientation(mc1gt_0,147,220). coords(mc1gt_0,151,128,220).  visible(mc1gt_0,220). happensAt(walking(mc1gt_0),220). orientation(mc1gt_2,152,220). coords(mc1gt_2,119,126,220).  visible(mc1gt_2,220). happensAt(walking(mc1gt_2),220).
orientation(mc1gt_0,147,221). coords(mc1gt_0,152,128,221).  visible(mc1gt_0,221). happensAt(walking(mc1gt_0),221). orientation(mc1gt_2,152,221). coords(mc1gt_2,121,127,221).  visible(mc1gt_2,221). happensAt(walking(mc1gt_2),221).
orientation(mc1gt_0,147,222). coords(mc1gt_0,152,129,222).  visible(mc1gt_0,222). happensAt(walking(mc1gt_0),222). orientation(mc1gt_2,152,222). coords(mc1gt_2,121,127,222).  visible(mc1gt_2,222). happensAt(walking(mc1gt_2),222).
orientation(mc1gt_0,147,223). coords(mc1gt_0,153,129,223).  visible(mc1gt_0,223). happensAt(walking(mc1gt_0),223). orientation(mc1gt_2,152,223). coords(mc1gt_2,122,127,223).  visible(mc1gt_2,223). happensAt(walking(mc1gt_2),223).
orientation(mc1gt_0,147,224). coords(mc1gt_0,153,130,224).  visible(mc1gt_0,224). happensAt(walking(mc1gt_0),224). orientation(mc1gt_2,149,224). coords(mc1gt_2,122,127,224).  visible(mc1gt_2,224). happensAt(walking(mc1gt_2),224).
orientation(mc1gt_0,147,225). coords(mc1gt_0,153,130,225).  visible(mc1gt_0,225). happensAt(walking(mc1gt_0),225). orientation(mc1gt_2,148,225). coords(mc1gt_2,123,127,225).  visible(mc1gt_2,225). happensAt(walking(mc1gt_2),225).
orientation(mc1gt_0,147,226). coords(mc1gt_0,154,130,226).  visible(mc1gt_0,226). happensAt(walking(mc1gt_0),226). orientation(mc1gt_2,147,226). coords(mc1gt_2,123,127,226).  visible(mc1gt_2,226). happensAt(walking(mc1gt_2),226).
orientation(mc1gt_0,147,227). coords(mc1gt_0,155,131,227).  visible(mc1gt_0,227). happensAt(walking(mc1gt_0),227). orientation(mc1gt_2,147,227). coords(mc1gt_2,123,127,227).  visible(mc1gt_2,227). happensAt(walking(mc1gt_2),227).
orientation(mc1gt_0,147,228). coords(mc1gt_0,157,131,228).  visible(mc1gt_0,228). happensAt(walking(mc1gt_0),228). orientation(mc1gt_2,147,228). coords(mc1gt_2,123,129,228).  visible(mc1gt_2,228). happensAt(walking(mc1gt_2),228).
orientation(mc1gt_0,147,229). coords(mc1gt_0,159,132,229).  visible(mc1gt_0,229). happensAt(walking(mc1gt_0),229). orientation(mc1gt_2,147,229). coords(mc1gt_2,124,130,229).  visible(mc1gt_2,229). happensAt(walking(mc1gt_2),229).
orientation(mc1gt_0,147,230). coords(mc1gt_0,160,133,230).  visible(mc1gt_0,230). happensAt(walking(mc1gt_0),230). orientation(mc1gt_2,147,230). coords(mc1gt_2,125,131,230).  visible(mc1gt_2,230). happensAt(walking(mc1gt_2),230).
orientation(mc1gt_0,147,231). coords(mc1gt_0,161,133,231).  visible(mc1gt_0,231). happensAt(walking(mc1gt_0),231). orientation(mc1gt_2,146,231). coords(mc1gt_2,127,132,231).  visible(mc1gt_2,231). happensAt(walking(mc1gt_2),231).
orientation(mc1gt_0,147,232). coords(mc1gt_0,162,135,232).  visible(mc1gt_0,232). happensAt(walking(mc1gt_0),232). orientation(mc1gt_2,145,232). coords(mc1gt_2,129,133,232).  visible(mc1gt_2,232). happensAt(walking(mc1gt_2),232).
orientation(mc1gt_0,147,233). coords(mc1gt_0,163,136,233).  visible(mc1gt_0,233). happensAt(walking(mc1gt_0),233). orientation(mc1gt_2,147,233). coords(mc1gt_2,130,134,233).  visible(mc1gt_2,233). happensAt(walking(mc1gt_2),233).
orientation(mc1gt_0,147,234). coords(mc1gt_0,164,137,234).  visible(mc1gt_0,234). happensAt(walking(mc1gt_0),234). orientation(mc1gt_2,147,234). coords(mc1gt_2,131,134,234).  visible(mc1gt_2,234). happensAt(walking(mc1gt_2),234).
orientation(mc1gt_0,147,235). coords(mc1gt_0,165,138,235).  visible(mc1gt_0,235). happensAt(walking(mc1gt_0),235). orientation(mc1gt_2,147,235). coords(mc1gt_2,131,135,235).  visible(mc1gt_2,235). happensAt(walking(mc1gt_2),235).
orientation(mc1gt_0,147,236). coords(mc1gt_0,165,138,236).  visible(mc1gt_0,236). happensAt(walking(mc1gt_0),236). orientation(mc1gt_2,146,236). coords(mc1gt_2,131,135,236).  visible(mc1gt_2,236). happensAt(walking(mc1gt_2),236).
orientation(mc1gt_0,147,237). coords(mc1gt_0,166,139,237).  visible(mc1gt_0,237). happensAt(walking(mc1gt_0),237). orientation(mc1gt_2,146,237). coords(mc1gt_2,132,135,237).  visible(mc1gt_2,237). happensAt(walking(mc1gt_2),237).
#show holdsAt/2.