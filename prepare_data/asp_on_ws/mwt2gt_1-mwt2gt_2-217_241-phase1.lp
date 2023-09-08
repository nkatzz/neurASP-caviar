
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

	orientation(mwt2gt_1,141,217). coords(mwt2gt_1,147,52,217).  visible(mwt2gt_1,217). happensAt(walking(mwt2gt_1),217). orientation(mwt2gt_2,168,217). coords(mwt2gt_2,231,207,217).  visible(mwt2gt_2,217). happensAt(walking(mwt2gt_2),217).
orientation(mwt2gt_1,141,218). coords(mwt2gt_1,146,54,218).  visible(mwt2gt_1,218). happensAt(walking(mwt2gt_1),218). orientation(mwt2gt_2,168,218). coords(mwt2gt_2,230,204,218).  visible(mwt2gt_2,218). happensAt(walking(mwt2gt_2),218).
orientation(mwt2gt_1,137,219). coords(mwt2gt_1,146,55,219).  visible(mwt2gt_1,219). happensAt(walking(mwt2gt_1),219). orientation(mwt2gt_2,168,219). coords(mwt2gt_2,228,202,219).  visible(mwt2gt_2,219). happensAt(walking(mwt2gt_2),219).
orientation(mwt2gt_1,133,220). coords(mwt2gt_1,146,56,220).  visible(mwt2gt_1,220). happensAt(walking(mwt2gt_1),220). orientation(mwt2gt_2,168,220). coords(mwt2gt_2,227,201,220).  visible(mwt2gt_2,220). happensAt(walking(mwt2gt_2),220).
orientation(mwt2gt_1,133,221). coords(mwt2gt_1,146,57,221).  visible(mwt2gt_1,221). happensAt(walking(mwt2gt_1),221). orientation(mwt2gt_2,168,221). coords(mwt2gt_2,227,198,221).  visible(mwt2gt_2,221). happensAt(walking(mwt2gt_2),221).
orientation(mwt2gt_1,133,222). coords(mwt2gt_1,145,58,222).  visible(mwt2gt_1,222). happensAt(walking(mwt2gt_1),222). orientation(mwt2gt_2,168,222). coords(mwt2gt_2,226,195,222).  visible(mwt2gt_2,222). happensAt(walking(mwt2gt_2),222).
orientation(mwt2gt_1,133,223). coords(mwt2gt_1,146,58,223).  visible(mwt2gt_1,223). happensAt(walking(mwt2gt_1),223). orientation(mwt2gt_2,168,223). coords(mwt2gt_2,226,194,223).  visible(mwt2gt_2,223). happensAt(walking(mwt2gt_2),223).
orientation(mwt2gt_1,133,224). coords(mwt2gt_1,145,59,224).  visible(mwt2gt_1,224). happensAt(walking(mwt2gt_1),224). orientation(mwt2gt_2,159,224). coords(mwt2gt_2,226,194,224).  visible(mwt2gt_2,224). happensAt(walking(mwt2gt_2),224).
orientation(mwt2gt_1,136,225). coords(mwt2gt_1,145,59,225).  visible(mwt2gt_1,225). happensAt(walking(mwt2gt_1),225). orientation(mwt2gt_2,150,225). coords(mwt2gt_2,225,194,225).  visible(mwt2gt_2,225). happensAt(walking(mwt2gt_2),225).
orientation(mwt2gt_1,136,226). coords(mwt2gt_1,144,59,226).  visible(mwt2gt_1,226). happensAt(walking(mwt2gt_1),226). orientation(mwt2gt_2,146,226). coords(mwt2gt_2,224,192,226).  visible(mwt2gt_2,226). happensAt(walking(mwt2gt_2),226).
orientation(mwt2gt_1,142,227). coords(mwt2gt_1,144,60,227).  visible(mwt2gt_1,227). happensAt(walking(mwt2gt_1),227). orientation(mwt2gt_2,146,227). coords(mwt2gt_2,223,190,227).  visible(mwt2gt_2,227). happensAt(walking(mwt2gt_2),227).
orientation(mwt2gt_1,142,228). coords(mwt2gt_1,144,60,228).  visible(mwt2gt_1,228). happensAt(walking(mwt2gt_1),228). orientation(mwt2gt_2,146,228). coords(mwt2gt_2,221,187,228).  visible(mwt2gt_2,228). happensAt(walking(mwt2gt_2),228).
orientation(mwt2gt_1,142,229). coords(mwt2gt_1,143,60,229).  visible(mwt2gt_1,229). happensAt(walking(mwt2gt_1),229). orientation(mwt2gt_2,146,229). coords(mwt2gt_2,220,184,229).  visible(mwt2gt_2,229). happensAt(walking(mwt2gt_2),229).
orientation(mwt2gt_1,142,230). coords(mwt2gt_1,142,61,230).  visible(mwt2gt_1,230). happensAt(walking(mwt2gt_1),230). orientation(mwt2gt_2,146,230). coords(mwt2gt_2,218,180,230).  visible(mwt2gt_2,230). happensAt(walking(mwt2gt_2),230).
orientation(mwt2gt_1,142,231). coords(mwt2gt_1,142,61,231).  visible(mwt2gt_1,231). happensAt(walking(mwt2gt_1),231). orientation(mwt2gt_2,146,231). coords(mwt2gt_2,218,180,231).  visible(mwt2gt_2,231). happensAt(walking(mwt2gt_2),231).
orientation(mwt2gt_1,142,232). coords(mwt2gt_1,142,61,232).  visible(mwt2gt_1,232). happensAt(walking(mwt2gt_1),232). orientation(mwt2gt_2,146,232). coords(mwt2gt_2,217,174,232).  visible(mwt2gt_2,232). happensAt(walking(mwt2gt_2),232).
orientation(mwt2gt_1,142,233). coords(mwt2gt_1,142,61,233).  visible(mwt2gt_1,233). happensAt(walking(mwt2gt_1),233). orientation(mwt2gt_2,146,233). coords(mwt2gt_2,216,174,233).  visible(mwt2gt_2,233). happensAt(walking(mwt2gt_2),233).
orientation(mwt2gt_1,142,234). coords(mwt2gt_1,141,62,234).  visible(mwt2gt_1,234). happensAt(walking(mwt2gt_1),234). orientation(mwt2gt_2,146,234). coords(mwt2gt_2,216,172,234).  visible(mwt2gt_2,234). happensAt(walking(mwt2gt_2),234).
orientation(mwt2gt_1,142,235). coords(mwt2gt_1,140,64,235).  visible(mwt2gt_1,235). happensAt(walking(mwt2gt_1),235). orientation(mwt2gt_2,146,235). coords(mwt2gt_2,215,171,235).  visible(mwt2gt_2,235). happensAt(walking(mwt2gt_2),235).
orientation(mwt2gt_1,142,236). coords(mwt2gt_1,140,65,236).  visible(mwt2gt_1,236). happensAt(walking(mwt2gt_1),236). orientation(mwt2gt_2,146,236). coords(mwt2gt_2,215,170,236).  visible(mwt2gt_2,236). happensAt(walking(mwt2gt_2),236).
orientation(mwt2gt_1,142,237). coords(mwt2gt_1,140,65,237).  visible(mwt2gt_1,237). happensAt(walking(mwt2gt_1),237). orientation(mwt2gt_2,146,237). coords(mwt2gt_2,214,169,237).  visible(mwt2gt_2,237). happensAt(walking(mwt2gt_2),237).
orientation(mwt2gt_1,142,238). coords(mwt2gt_1,139,66,238).  visible(mwt2gt_1,238). happensAt(walking(mwt2gt_1),238). orientation(mwt2gt_2,146,238). coords(mwt2gt_2,213,168,238).  visible(mwt2gt_2,238). happensAt(walking(mwt2gt_2),238).
orientation(mwt2gt_1,142,239). coords(mwt2gt_1,140,67,239).  visible(mwt2gt_1,239). happensAt(walking(mwt2gt_1),239). orientation(mwt2gt_2,146,239). coords(mwt2gt_2,213,167,239).  visible(mwt2gt_2,239). happensAt(walking(mwt2gt_2),239).
orientation(mwt2gt_1,138,240). coords(mwt2gt_1,140,67,240).  visible(mwt2gt_1,240). happensAt(walking(mwt2gt_1),240). orientation(mwt2gt_2,146,240). coords(mwt2gt_2,211,165,240).  visible(mwt2gt_2,240). happensAt(walking(mwt2gt_2),240).
orientation(mwt2gt_1,138,241). coords(mwt2gt_1,140,68,241).  visible(mwt2gt_1,241). happensAt(walking(mwt2gt_1),241). orientation(mwt2gt_2,146,241). coords(mwt2gt_2,210,163,241).  visible(mwt2gt_2,241). happensAt(walking(mwt2gt_2),241).
#show holdsAt/2.