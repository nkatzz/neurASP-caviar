
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

	orientation(fcgt_1,171,220). coords(fcgt_1,210,212,220).  visible(fcgt_1,220). happensAt(walking(fcgt_1),220). orientation(fcgt_3,149,220). coords(fcgt_3,100,170,220).  visible(fcgt_3,220). happensAt(walking(fcgt_3),220).
orientation(fcgt_1,171,221). coords(fcgt_1,210,210,221).  visible(fcgt_1,221). happensAt(walking(fcgt_1),221). orientation(fcgt_3,149,221). coords(fcgt_3,100,170,221).  visible(fcgt_3,221). happensAt(walking(fcgt_3),221).
orientation(fcgt_1,171,222). coords(fcgt_1,209,209,222).  visible(fcgt_1,222). happensAt(walking(fcgt_1),222). orientation(fcgt_3,149,222). coords(fcgt_3,100,170,222).  visible(fcgt_3,222). happensAt(walking(fcgt_3),222).
orientation(fcgt_1,171,223). coords(fcgt_1,206,206,223).  visible(fcgt_1,223). happensAt(walking(fcgt_1),223). orientation(fcgt_3,149,223). coords(fcgt_3,100,170,223).  visible(fcgt_3,223). happensAt(walking(fcgt_3),223).
orientation(fcgt_1,171,224). coords(fcgt_1,205,204,224).  visible(fcgt_1,224). happensAt(walking(fcgt_1),224). orientation(fcgt_3,149,224). coords(fcgt_3,100,170,224).  visible(fcgt_3,224). happensAt(walking(fcgt_3),224).
orientation(fcgt_1,171,225). coords(fcgt_1,203,202,225).  visible(fcgt_1,225). happensAt(walking(fcgt_1),225). orientation(fcgt_3,149,225). coords(fcgt_3,100,171,225).  visible(fcgt_3,225). happensAt(walking(fcgt_3),225).
orientation(fcgt_1,171,226). coords(fcgt_1,200,199,226).  visible(fcgt_1,226). happensAt(walking(fcgt_1),226). orientation(fcgt_3,149,226). coords(fcgt_3,100,171,226).  visible(fcgt_3,226). happensAt(walking(fcgt_3),226).
orientation(fcgt_1,171,227). coords(fcgt_1,197,197,227).  visible(fcgt_1,227). happensAt(walking(fcgt_1),227). orientation(fcgt_3,157,227). coords(fcgt_3,100,171,227).  visible(fcgt_3,227). happensAt(walking(fcgt_3),227).
orientation(fcgt_1,171,228). coords(fcgt_1,194,194,228).  visible(fcgt_1,228). happensAt(walking(fcgt_1),228). orientation(fcgt_3,159,228). coords(fcgt_3,100,171,228).  visible(fcgt_3,228). happensAt(walking(fcgt_3),228).
orientation(fcgt_1,167,229). coords(fcgt_1,191,194,229).  visible(fcgt_1,229). happensAt(walking(fcgt_1),229). orientation(fcgt_3,159,229). coords(fcgt_3,100,172,229).  visible(fcgt_3,229). happensAt(walking(fcgt_3),229).
orientation(fcgt_1,163,230). coords(fcgt_1,190,193,230).  visible(fcgt_1,230). happensAt(walking(fcgt_1),230). orientation(fcgt_3,159,230). coords(fcgt_3,100,172,230).  visible(fcgt_3,230). happensAt(walking(fcgt_3),230).
orientation(fcgt_1,163,231). coords(fcgt_1,189,192,231).  visible(fcgt_1,231). happensAt(walking(fcgt_1),231). orientation(fcgt_3,159,231). coords(fcgt_3,100,172,231).  visible(fcgt_3,231). happensAt(walking(fcgt_3),231).
orientation(fcgt_1,163,232). coords(fcgt_1,189,190,232).  visible(fcgt_1,232). happensAt(walking(fcgt_1),232). orientation(fcgt_3,159,232). coords(fcgt_3,100,172,232).  visible(fcgt_3,232). happensAt(walking(fcgt_3),232).
orientation(fcgt_1,156,233). coords(fcgt_1,188,188,233).  visible(fcgt_1,233). happensAt(walking(fcgt_1),233). orientation(fcgt_3,159,233). coords(fcgt_3,100,172,233).  visible(fcgt_3,233). happensAt(walking(fcgt_3),233).
orientation(fcgt_1,156,234). coords(fcgt_1,186,188,234).  visible(fcgt_1,234). happensAt(walking(fcgt_1),234). orientation(fcgt_3,159,234). coords(fcgt_3,100,172,234).  visible(fcgt_3,234). happensAt(walking(fcgt_3),234).
orientation(fcgt_1,149,235). coords(fcgt_1,186,186,235).  visible(fcgt_1,235). happensAt(walking(fcgt_1),235). orientation(fcgt_3,159,235). coords(fcgt_3,100,173,235).  visible(fcgt_3,235). happensAt(walking(fcgt_3),235).
orientation(fcgt_1,149,236). coords(fcgt_1,184,183,236).  visible(fcgt_1,236). happensAt(walking(fcgt_1),236). orientation(fcgt_3,159,236). coords(fcgt_3,100,173,236).  visible(fcgt_3,236). happensAt(walking(fcgt_3),236).
orientation(fcgt_1,149,237). coords(fcgt_1,181,181,237).  visible(fcgt_1,237). happensAt(walking(fcgt_1),237). orientation(fcgt_3,159,237). coords(fcgt_3,100,173,237).  visible(fcgt_3,237). happensAt(walking(fcgt_3),237).
orientation(fcgt_1,149,238). coords(fcgt_1,177,178,238).  visible(fcgt_1,238). happensAt(walking(fcgt_1),238). orientation(fcgt_3,159,238). coords(fcgt_3,100,174,238).  visible(fcgt_3,238). happensAt(walking(fcgt_3),238).
orientation(fcgt_1,149,239). coords(fcgt_1,175,175,239).  visible(fcgt_1,239). happensAt(walking(fcgt_1),239). orientation(fcgt_3,163,239). coords(fcgt_3,101,174,239).  visible(fcgt_3,239). happensAt(walking(fcgt_3),239).
orientation(fcgt_1,149,240). coords(fcgt_1,173,171,240).  visible(fcgt_1,240). happensAt(walking(fcgt_1),240). orientation(fcgt_3,166,240). coords(fcgt_3,101,175,240).  visible(fcgt_3,240). happensAt(walking(fcgt_3),240).
orientation(fcgt_1,149,241). coords(fcgt_1,173,170,241).  visible(fcgt_1,241). happensAt(walking(fcgt_1),241). orientation(fcgt_3,167,241). coords(fcgt_3,101,175,241).  visible(fcgt_3,241). happensAt(walking(fcgt_3),241).
orientation(fcgt_1,149,242). coords(fcgt_1,172,165,242).  visible(fcgt_1,242). happensAt(walking(fcgt_1),242). orientation(fcgt_3,166,242). coords(fcgt_3,101,175,242).  visible(fcgt_3,242). happensAt(walking(fcgt_3),242).
orientation(fcgt_1,149,243). coords(fcgt_1,171,163,243).  visible(fcgt_1,243). happensAt(walking(fcgt_1),243). orientation(fcgt_3,167,243). coords(fcgt_3,101,175,243).  visible(fcgt_3,243). happensAt(walking(fcgt_3),243).
orientation(fcgt_1,154,244). coords(fcgt_1,171,162,244).  visible(fcgt_1,244). happensAt(walking(fcgt_1),244). orientation(fcgt_3,170,244). coords(fcgt_3,101,175,244).  visible(fcgt_3,244). happensAt(walking(fcgt_3),244).
#show holdsAt/2.