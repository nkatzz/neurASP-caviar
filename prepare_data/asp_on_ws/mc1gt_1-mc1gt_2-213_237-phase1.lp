
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

	orientation(mc1gt_1,143,213). coords(mc1gt_1,167,112,213).  visible(mc1gt_1,213). happensAt(walking(mc1gt_1),213). orientation(mc1gt_2,148,213). coords(mc1gt_2,111,125,213).  visible(mc1gt_2,213). happensAt(walking(mc1gt_2),213).
orientation(mc1gt_1,143,214). coords(mc1gt_1,169,113,214).  visible(mc1gt_1,214). happensAt(walking(mc1gt_1),214). orientation(mc1gt_2,148,214). coords(mc1gt_2,111,125,214).  visible(mc1gt_2,214). happensAt(walking(mc1gt_2),214).
orientation(mc1gt_1,143,215). coords(mc1gt_1,171,114,215).  visible(mc1gt_1,215). happensAt(walking(mc1gt_1),215). orientation(mc1gt_2,152,215). coords(mc1gt_2,113,125,215).  visible(mc1gt_2,215). happensAt(walking(mc1gt_2),215).
orientation(mc1gt_1,142,216). coords(mc1gt_1,173,115,216).  visible(mc1gt_1,216). happensAt(walking(mc1gt_1),216). orientation(mc1gt_2,152,216). coords(mc1gt_2,115,125,216).  visible(mc1gt_2,216). happensAt(walking(mc1gt_2),216).
orientation(mc1gt_1,141,217). coords(mc1gt_1,174,116,217).  visible(mc1gt_1,217). happensAt(walking(mc1gt_1),217). orientation(mc1gt_2,152,217). coords(mc1gt_2,116,125,217).  visible(mc1gt_2,217). happensAt(walking(mc1gt_2),217).
orientation(mc1gt_1,141,218). coords(mc1gt_1,175,117,218).  visible(mc1gt_1,218). happensAt(walking(mc1gt_1),218). orientation(mc1gt_2,152,218). coords(mc1gt_2,117,125,218).  visible(mc1gt_2,218). happensAt(walking(mc1gt_2),218).
orientation(mc1gt_1,141,219). coords(mc1gt_1,176,118,219).  visible(mc1gt_1,219). happensAt(walking(mc1gt_1),219). orientation(mc1gt_2,152,219). coords(mc1gt_2,119,126,219).  visible(mc1gt_2,219). happensAt(walking(mc1gt_2),219).
orientation(mc1gt_1,141,220). coords(mc1gt_1,177,119,220).  visible(mc1gt_1,220). happensAt(walking(mc1gt_1),220). orientation(mc1gt_2,152,220). coords(mc1gt_2,119,126,220).  visible(mc1gt_2,220). happensAt(walking(mc1gt_2),220).
orientation(mc1gt_1,139,221). coords(mc1gt_1,178,119,221).  visible(mc1gt_1,221). happensAt(walking(mc1gt_1),221). orientation(mc1gt_2,152,221). coords(mc1gt_2,121,127,221).  visible(mc1gt_2,221). happensAt(walking(mc1gt_2),221).
orientation(mc1gt_1,138,222). coords(mc1gt_1,178,120,222).  visible(mc1gt_1,222). happensAt(walking(mc1gt_1),222). orientation(mc1gt_2,152,222). coords(mc1gt_2,121,127,222).  visible(mc1gt_2,222). happensAt(walking(mc1gt_2),222).
orientation(mc1gt_1,138,223). coords(mc1gt_1,178,120,223).  visible(mc1gt_1,223). happensAt(walking(mc1gt_1),223). orientation(mc1gt_2,152,223). coords(mc1gt_2,122,127,223).  visible(mc1gt_2,223). happensAt(walking(mc1gt_2),223).
orientation(mc1gt_1,138,224). coords(mc1gt_1,179,120,224).  visible(mc1gt_1,224). happensAt(walking(mc1gt_1),224). orientation(mc1gt_2,149,224). coords(mc1gt_2,122,127,224).  visible(mc1gt_2,224). happensAt(walking(mc1gt_2),224).
orientation(mc1gt_1,139,225). coords(mc1gt_1,179,121,225).  visible(mc1gt_1,225). happensAt(walking(mc1gt_1),225). orientation(mc1gt_2,148,225). coords(mc1gt_2,123,127,225).  visible(mc1gt_2,225). happensAt(walking(mc1gt_2),225).
orientation(mc1gt_1,142,226). coords(mc1gt_1,179,121,226).  visible(mc1gt_1,226). happensAt(walking(mc1gt_1),226). orientation(mc1gt_2,147,226). coords(mc1gt_2,123,127,226).  visible(mc1gt_2,226). happensAt(walking(mc1gt_2),226).
orientation(mc1gt_1,142,227). coords(mc1gt_1,179,123,227).  visible(mc1gt_1,227). happensAt(walking(mc1gt_1),227). orientation(mc1gt_2,147,227). coords(mc1gt_2,123,127,227).  visible(mc1gt_2,227). happensAt(walking(mc1gt_2),227).
orientation(mc1gt_1,142,228). coords(mc1gt_1,179,126,228).  visible(mc1gt_1,228). happensAt(walking(mc1gt_1),228). orientation(mc1gt_2,147,228). coords(mc1gt_2,123,129,228).  visible(mc1gt_2,228). happensAt(walking(mc1gt_2),228).
orientation(mc1gt_1,142,229). coords(mc1gt_1,181,129,229).  visible(mc1gt_1,229). happensAt(walking(mc1gt_1),229). orientation(mc1gt_2,147,229). coords(mc1gt_2,124,130,229).  visible(mc1gt_2,229). happensAt(walking(mc1gt_2),229).
orientation(mc1gt_1,142,230). coords(mc1gt_1,183,131,230).  visible(mc1gt_1,230). happensAt(walking(mc1gt_1),230). orientation(mc1gt_2,147,230). coords(mc1gt_2,125,131,230).  visible(mc1gt_2,230). happensAt(walking(mc1gt_2),230).
orientation(mc1gt_1,142,231). coords(mc1gt_1,184,133,231).  visible(mc1gt_1,231). happensAt(walking(mc1gt_1),231). orientation(mc1gt_2,146,231). coords(mc1gt_2,127,132,231).  visible(mc1gt_2,231). happensAt(walking(mc1gt_2),231).
orientation(mc1gt_1,142,232). coords(mc1gt_1,185,135,232).  visible(mc1gt_1,232). happensAt(walking(mc1gt_1),232). orientation(mc1gt_2,145,232). coords(mc1gt_2,129,133,232).  visible(mc1gt_2,232). happensAt(walking(mc1gt_2),232).
orientation(mc1gt_1,142,233). coords(mc1gt_1,186,137,233).  visible(mc1gt_1,233). happensAt(walking(mc1gt_1),233). orientation(mc1gt_2,147,233). coords(mc1gt_2,130,134,233).  visible(mc1gt_2,233). happensAt(walking(mc1gt_2),233).
orientation(mc1gt_1,142,234). coords(mc1gt_1,187,137,234).  visible(mc1gt_1,234). happensAt(walking(mc1gt_1),234). orientation(mc1gt_2,147,234). coords(mc1gt_2,131,134,234).  visible(mc1gt_2,234). happensAt(walking(mc1gt_2),234).
orientation(mc1gt_1,142,235). coords(mc1gt_1,188,138,235).  visible(mc1gt_1,235). happensAt(walking(mc1gt_1),235). orientation(mc1gt_2,147,235). coords(mc1gt_2,131,135,235).  visible(mc1gt_2,235). happensAt(walking(mc1gt_2),235).
orientation(mc1gt_1,142,236). coords(mc1gt_1,189,139,236).  visible(mc1gt_1,236). happensAt(walking(mc1gt_1),236). orientation(mc1gt_2,146,236). coords(mc1gt_2,131,135,236).  visible(mc1gt_2,236). happensAt(walking(mc1gt_2),236).
orientation(mc1gt_1,142,237). coords(mc1gt_1,189,140,237).  visible(mc1gt_1,237). happensAt(walking(mc1gt_1),237). orientation(mc1gt_2,146,237). coords(mc1gt_2,132,135,237).  visible(mc1gt_2,237). happensAt(walking(mc1gt_2),237).
#show holdsAt/2.