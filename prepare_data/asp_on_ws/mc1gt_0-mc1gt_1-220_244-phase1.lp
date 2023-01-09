
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

	orientation(mc1gt_0,147,220). coords(mc1gt_0,151,128,220).  visible(mc1gt_0,220). happensAt(walking(mc1gt_0),220). orientation(mc1gt_1,141,220). coords(mc1gt_1,177,119,220).  visible(mc1gt_1,220). happensAt(walking(mc1gt_1),220).
orientation(mc1gt_0,147,221). coords(mc1gt_0,152,128,221).  visible(mc1gt_0,221). happensAt(walking(mc1gt_0),221). orientation(mc1gt_1,139,221). coords(mc1gt_1,178,119,221).  visible(mc1gt_1,221). happensAt(walking(mc1gt_1),221).
orientation(mc1gt_0,147,222). coords(mc1gt_0,152,129,222).  visible(mc1gt_0,222). happensAt(walking(mc1gt_0),222). orientation(mc1gt_1,138,222). coords(mc1gt_1,178,120,222).  visible(mc1gt_1,222). happensAt(walking(mc1gt_1),222).
orientation(mc1gt_0,147,223). coords(mc1gt_0,153,129,223).  visible(mc1gt_0,223). happensAt(walking(mc1gt_0),223). orientation(mc1gt_1,138,223). coords(mc1gt_1,178,120,223).  visible(mc1gt_1,223). happensAt(walking(mc1gt_1),223).
orientation(mc1gt_0,147,224). coords(mc1gt_0,153,130,224).  visible(mc1gt_0,224). happensAt(walking(mc1gt_0),224). orientation(mc1gt_1,138,224). coords(mc1gt_1,179,120,224).  visible(mc1gt_1,224). happensAt(walking(mc1gt_1),224).
orientation(mc1gt_0,147,225). coords(mc1gt_0,153,130,225).  visible(mc1gt_0,225). happensAt(walking(mc1gt_0),225). orientation(mc1gt_1,139,225). coords(mc1gt_1,179,121,225).  visible(mc1gt_1,225). happensAt(walking(mc1gt_1),225).
orientation(mc1gt_0,147,226). coords(mc1gt_0,154,130,226).  visible(mc1gt_0,226). happensAt(walking(mc1gt_0),226). orientation(mc1gt_1,142,226). coords(mc1gt_1,179,121,226).  visible(mc1gt_1,226). happensAt(walking(mc1gt_1),226).
orientation(mc1gt_0,147,227). coords(mc1gt_0,155,131,227).  visible(mc1gt_0,227). happensAt(walking(mc1gt_0),227). orientation(mc1gt_1,142,227). coords(mc1gt_1,179,123,227).  visible(mc1gt_1,227). happensAt(walking(mc1gt_1),227).
orientation(mc1gt_0,147,228). coords(mc1gt_0,157,131,228).  visible(mc1gt_0,228). happensAt(walking(mc1gt_0),228). orientation(mc1gt_1,142,228). coords(mc1gt_1,179,126,228).  visible(mc1gt_1,228). happensAt(walking(mc1gt_1),228).
orientation(mc1gt_0,147,229). coords(mc1gt_0,159,132,229).  visible(mc1gt_0,229). happensAt(walking(mc1gt_0),229). orientation(mc1gt_1,142,229). coords(mc1gt_1,181,129,229).  visible(mc1gt_1,229). happensAt(walking(mc1gt_1),229).
orientation(mc1gt_0,147,230). coords(mc1gt_0,160,133,230).  visible(mc1gt_0,230). happensAt(walking(mc1gt_0),230). orientation(mc1gt_1,142,230). coords(mc1gt_1,183,131,230).  visible(mc1gt_1,230). happensAt(walking(mc1gt_1),230).
orientation(mc1gt_0,147,231). coords(mc1gt_0,161,133,231).  visible(mc1gt_0,231). happensAt(walking(mc1gt_0),231). orientation(mc1gt_1,142,231). coords(mc1gt_1,184,133,231).  visible(mc1gt_1,231). happensAt(walking(mc1gt_1),231).
orientation(mc1gt_0,147,232). coords(mc1gt_0,162,135,232).  visible(mc1gt_0,232). happensAt(walking(mc1gt_0),232). orientation(mc1gt_1,142,232). coords(mc1gt_1,185,135,232).  visible(mc1gt_1,232). happensAt(walking(mc1gt_1),232).
orientation(mc1gt_0,147,233). coords(mc1gt_0,163,136,233).  visible(mc1gt_0,233). happensAt(walking(mc1gt_0),233). orientation(mc1gt_1,142,233). coords(mc1gt_1,186,137,233).  visible(mc1gt_1,233). happensAt(walking(mc1gt_1),233).
orientation(mc1gt_0,147,234). coords(mc1gt_0,164,137,234).  visible(mc1gt_0,234). happensAt(walking(mc1gt_0),234). orientation(mc1gt_1,142,234). coords(mc1gt_1,187,137,234).  visible(mc1gt_1,234). happensAt(walking(mc1gt_1),234).
orientation(mc1gt_0,147,235). coords(mc1gt_0,165,138,235).  visible(mc1gt_0,235). happensAt(walking(mc1gt_0),235). orientation(mc1gt_1,142,235). coords(mc1gt_1,188,138,235).  visible(mc1gt_1,235). happensAt(walking(mc1gt_1),235).
orientation(mc1gt_0,147,236). coords(mc1gt_0,165,138,236).  visible(mc1gt_0,236). happensAt(walking(mc1gt_0),236). orientation(mc1gt_1,142,236). coords(mc1gt_1,189,139,236).  visible(mc1gt_1,236). happensAt(walking(mc1gt_1),236).
orientation(mc1gt_0,147,237). coords(mc1gt_0,166,139,237).  visible(mc1gt_0,237). happensAt(walking(mc1gt_0),237). orientation(mc1gt_1,142,237). coords(mc1gt_1,189,140,237).  visible(mc1gt_1,237). happensAt(walking(mc1gt_1),237).
orientation(mc1gt_0,147,238). coords(mc1gt_0,166,139,238).  visible(mc1gt_0,238). happensAt(walking(mc1gt_0),238). orientation(mc1gt_1,142,238). coords(mc1gt_1,190,141,238).  visible(mc1gt_1,238). happensAt(walking(mc1gt_1),238).
orientation(mc1gt_0,147,239). coords(mc1gt_0,166,140,239).  visible(mc1gt_0,239). happensAt(walking(mc1gt_0),239). orientation(mc1gt_1,142,239). coords(mc1gt_1,190,141,239).  visible(mc1gt_1,239). happensAt(walking(mc1gt_1),239).
orientation(mc1gt_0,147,240). coords(mc1gt_0,166,140,240).  visible(mc1gt_0,240). happensAt(walking(mc1gt_0),240). orientation(mc1gt_1,145,240). coords(mc1gt_1,190,142,240).  visible(mc1gt_1,240). happensAt(walking(mc1gt_1),240).
orientation(mc1gt_0,147,241). coords(mc1gt_0,167,143,241).  visible(mc1gt_0,241). happensAt(walking(mc1gt_0),241). orientation(mc1gt_1,147,241). coords(mc1gt_1,191,142,241).  visible(mc1gt_1,241). happensAt(walking(mc1gt_1),241).
orientation(mc1gt_0,147,242). coords(mc1gt_0,167,146,242).  visible(mc1gt_0,242). happensAt(walking(mc1gt_0),242). orientation(mc1gt_1,146,242). coords(mc1gt_1,192,143,242).  visible(mc1gt_1,242). happensAt(walking(mc1gt_1),242).
orientation(mc1gt_0,147,243). coords(mc1gt_0,167,149,243).  visible(mc1gt_0,243). happensAt(walking(mc1gt_0),243). orientation(mc1gt_1,148,243). coords(mc1gt_1,194,144,243).  visible(mc1gt_1,243). happensAt(walking(mc1gt_1),243).
orientation(mc1gt_0,147,244). coords(mc1gt_0,168,151,244).  visible(mc1gt_0,244). happensAt(walking(mc1gt_0),244). orientation(mc1gt_1,145,244). coords(mc1gt_1,197,145,244).  visible(mc1gt_1,244). happensAt(walking(mc1gt_1),244).
#show holdsAt/2.