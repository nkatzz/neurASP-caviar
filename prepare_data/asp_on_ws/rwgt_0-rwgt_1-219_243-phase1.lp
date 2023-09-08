
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

	orientation(rwgt_0,144,219). coords(rwgt_0,213,153,219).  visible(rwgt_0,219). happensAt(walking(rwgt_0),219). orientation(rwgt_1,132,219). coords(rwgt_1,155,34,219).  visible(rwgt_1,219). happensAt(walking(rwgt_1),219).
orientation(rwgt_0,144,220). coords(rwgt_0,213,153,220).  visible(rwgt_0,220). happensAt(walking(rwgt_0),220). orientation(rwgt_1,132,220). coords(rwgt_1,156,34,220).  visible(rwgt_1,220). happensAt(walking(rwgt_1),220).
orientation(rwgt_0,144,221). coords(rwgt_0,214,155,221).  visible(rwgt_0,221). happensAt(walking(rwgt_0),221). orientation(rwgt_1,132,221). coords(rwgt_1,158,34,221).  visible(rwgt_1,221). happensAt(walking(rwgt_1),221).
orientation(rwgt_0,144,222). coords(rwgt_0,215,156,222).  visible(rwgt_0,222). happensAt(walking(rwgt_0),222). orientation(rwgt_1,132,222). coords(rwgt_1,159,34,222).  visible(rwgt_1,222). happensAt(walking(rwgt_1),222).
orientation(rwgt_0,144,223). coords(rwgt_0,216,158,223).  visible(rwgt_0,223). happensAt(walking(rwgt_0),223). orientation(rwgt_1,132,223). coords(rwgt_1,159,34,223).  visible(rwgt_1,223). happensAt(walking(rwgt_1),223).
orientation(rwgt_0,144,224). coords(rwgt_0,218,162,224).  visible(rwgt_0,224). happensAt(walking(rwgt_0),224). orientation(rwgt_1,132,224). coords(rwgt_1,159,33,224).  visible(rwgt_1,224). happensAt(walking(rwgt_1),224).
orientation(rwgt_0,144,225). coords(rwgt_0,220,165,225).  visible(rwgt_0,225). happensAt(walking(rwgt_0),225). orientation(rwgt_1,132,225). coords(rwgt_1,158,32,225).  visible(rwgt_1,225). happensAt(walking(rwgt_1),225).
orientation(rwgt_0,144,226). coords(rwgt_0,223,168,226).  visible(rwgt_0,226). happensAt(walking(rwgt_0),226). orientation(rwgt_1,132,226). coords(rwgt_1,158,31,226).  visible(rwgt_1,226). happensAt(walking(rwgt_1),226).
orientation(rwgt_0,144,227). coords(rwgt_0,225,170,227).  visible(rwgt_0,227). happensAt(walking(rwgt_0),227). orientation(rwgt_1,132,227). coords(rwgt_1,158,31,227).  visible(rwgt_1,227). happensAt(walking(rwgt_1),227).
orientation(rwgt_0,144,228). coords(rwgt_0,228,172,228).  visible(rwgt_0,228). happensAt(walking(rwgt_0),228). orientation(rwgt_1,132,228). coords(rwgt_1,159,31,228).  visible(rwgt_1,228). happensAt(walking(rwgt_1),228).
orientation(rwgt_0,144,229). coords(rwgt_0,230,174,229).  visible(rwgt_0,229). happensAt(walking(rwgt_0),229). orientation(rwgt_1,132,229). coords(rwgt_1,159,31,229).  visible(rwgt_1,229). happensAt(walking(rwgt_1),229).
orientation(rwgt_0,144,230). coords(rwgt_0,231,175,230).  visible(rwgt_0,230). happensAt(walking(rwgt_0),230). orientation(rwgt_1,132,230). coords(rwgt_1,160,31,230).  visible(rwgt_1,230). happensAt(walking(rwgt_1),230).
orientation(rwgt_0,144,231). coords(rwgt_0,233,176,231).  visible(rwgt_0,231). happensAt(walking(rwgt_0),231). orientation(rwgt_1,132,231). coords(rwgt_1,160,31,231).  visible(rwgt_1,231). happensAt(walking(rwgt_1),231).
orientation(rwgt_0,144,232). coords(rwgt_0,233,177,232).  visible(rwgt_0,232). happensAt(walking(rwgt_0),232). orientation(rwgt_1,132,232). coords(rwgt_1,160,31,232).  visible(rwgt_1,232). happensAt(walking(rwgt_1),232).
orientation(rwgt_0,144,233). coords(rwgt_0,234,178,233).  visible(rwgt_0,233). happensAt(walking(rwgt_0),233). orientation(rwgt_1,132,233). coords(rwgt_1,160,31,233).  visible(rwgt_1,233). happensAt(walking(rwgt_1),233).
orientation(rwgt_0,148,234). coords(rwgt_0,235,179,234).  visible(rwgt_0,234). happensAt(walking(rwgt_0),234). orientation(rwgt_1,132,234). coords(rwgt_1,162,29,234).  visible(rwgt_1,234). happensAt(walking(rwgt_1),234).
orientation(rwgt_0,148,235). coords(rwgt_0,236,180,235).  visible(rwgt_0,235). happensAt(walking(rwgt_0),235). orientation(rwgt_1,132,235). coords(rwgt_1,163,29,235).  visible(rwgt_1,235). happensAt(walking(rwgt_1),235).
orientation(rwgt_0,150,236). coords(rwgt_0,237,181,236).  visible(rwgt_0,236). happensAt(walking(rwgt_0),236). orientation(rwgt_1,132,236). coords(rwgt_1,164,29,236).  visible(rwgt_1,236). happensAt(walking(rwgt_1),236).
orientation(rwgt_0,150,237). coords(rwgt_0,240,182,237).  visible(rwgt_0,237). happensAt(walking(rwgt_0),237). orientation(rwgt_1,132,237). coords(rwgt_1,164,29,237).  visible(rwgt_1,237). happensAt(walking(rwgt_1),237).
orientation(rwgt_0,150,238). coords(rwgt_0,243,184,238).  visible(rwgt_0,238). happensAt(walking(rwgt_0),238). orientation(rwgt_1,132,238). coords(rwgt_1,164,28,238).  visible(rwgt_1,238). happensAt(walking(rwgt_1),238).
orientation(rwgt_0,150,239). coords(rwgt_0,247,186,239).  visible(rwgt_0,239). happensAt(walking(rwgt_0),239). orientation(rwgt_1,132,239). coords(rwgt_1,165,27,239).  visible(rwgt_1,239). happensAt(walking(rwgt_1),239).
orientation(rwgt_0,150,240). coords(rwgt_0,251,188,240).  visible(rwgt_0,240). happensAt(walking(rwgt_0),240). orientation(rwgt_1,132,240). coords(rwgt_1,165,27,240).  visible(rwgt_1,240). happensAt(walking(rwgt_1),240).
orientation(rwgt_0,150,241). coords(rwgt_0,254,189,241).  visible(rwgt_0,241). happensAt(walking(rwgt_0),241). orientation(rwgt_1,132,241). coords(rwgt_1,165,27,241).  visible(rwgt_1,241). happensAt(walking(rwgt_1),241).
orientation(rwgt_0,150,242). coords(rwgt_0,257,190,242).  visible(rwgt_0,242). happensAt(walking(rwgt_0),242). orientation(rwgt_1,132,242). coords(rwgt_1,165,27,242).  visible(rwgt_1,242). happensAt(walking(rwgt_1),242).
orientation(rwgt_0,150,243). coords(rwgt_0,258,192,243).  visible(rwgt_0,243). happensAt(walking(rwgt_0),243). orientation(rwgt_1,132,243). coords(rwgt_1,166,26,243).  visible(rwgt_1,243). happensAt(walking(rwgt_1),243).
#show holdsAt/2.