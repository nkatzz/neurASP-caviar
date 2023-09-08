
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

	orientation(fcgt_1,171,220). coords(fcgt_1,210,212,220).  visible(fcgt_1,220). happensAt(walking(fcgt_1),220). orientation(fcgt_4,172,220). coords(fcgt_4,81,205,220).  visible(fcgt_4,220). happensAt(walking(fcgt_4),220).
orientation(fcgt_1,171,221). coords(fcgt_1,210,210,221).  visible(fcgt_1,221). happensAt(walking(fcgt_1),221). orientation(fcgt_4,172,221). coords(fcgt_4,81,205,221).  visible(fcgt_4,221). happensAt(walking(fcgt_4),221).
orientation(fcgt_1,171,222). coords(fcgt_1,209,209,222).  visible(fcgt_1,222). happensAt(walking(fcgt_1),222). orientation(fcgt_4,172,222). coords(fcgt_4,81,205,222).  visible(fcgt_4,222). happensAt(walking(fcgt_4),222).
orientation(fcgt_1,171,223). coords(fcgt_1,206,206,223).  visible(fcgt_1,223). happensAt(walking(fcgt_1),223). orientation(fcgt_4,166,223). coords(fcgt_4,81,205,223).  visible(fcgt_4,223). happensAt(inactive(fcgt_4),223).
orientation(fcgt_1,171,224). coords(fcgt_1,205,204,224).  visible(fcgt_1,224). happensAt(walking(fcgt_1),224). orientation(fcgt_4,166,224). coords(fcgt_4,81,205,224).  visible(fcgt_4,224). happensAt(inactive(fcgt_4),224).
orientation(fcgt_1,171,225). coords(fcgt_1,203,202,225).  visible(fcgt_1,225). happensAt(walking(fcgt_1),225). orientation(fcgt_4,166,225). coords(fcgt_4,80,205,225).  visible(fcgt_4,225). happensAt(inactive(fcgt_4),225).
orientation(fcgt_1,171,226). coords(fcgt_1,200,199,226).  visible(fcgt_1,226). happensAt(walking(fcgt_1),226). orientation(fcgt_4,166,226). coords(fcgt_4,80,205,226).  visible(fcgt_4,226). happensAt(inactive(fcgt_4),226).
orientation(fcgt_1,171,227). coords(fcgt_1,197,197,227).  visible(fcgt_1,227). happensAt(walking(fcgt_1),227). orientation(fcgt_4,166,227). coords(fcgt_4,80,205,227).  visible(fcgt_4,227). happensAt(inactive(fcgt_4),227).
orientation(fcgt_1,171,228). coords(fcgt_1,194,194,228).  visible(fcgt_1,228). happensAt(walking(fcgt_1),228). orientation(fcgt_4,166,228). coords(fcgt_4,80,205,228).  visible(fcgt_4,228). happensAt(inactive(fcgt_4),228).
orientation(fcgt_1,167,229). coords(fcgt_1,191,194,229).  visible(fcgt_1,229). happensAt(walking(fcgt_1),229). orientation(fcgt_4,166,229). coords(fcgt_4,80,205,229).  visible(fcgt_4,229). happensAt(inactive(fcgt_4),229).
orientation(fcgt_1,163,230). coords(fcgt_1,190,193,230).  visible(fcgt_1,230). happensAt(walking(fcgt_1),230). orientation(fcgt_4,166,230). coords(fcgt_4,80,205,230).  visible(fcgt_4,230). happensAt(inactive(fcgt_4),230).
orientation(fcgt_1,163,231). coords(fcgt_1,189,192,231).  visible(fcgt_1,231). happensAt(walking(fcgt_1),231). orientation(fcgt_4,166,231). coords(fcgt_4,80,205,231).  visible(fcgt_4,231). happensAt(inactive(fcgt_4),231).
orientation(fcgt_1,163,232). coords(fcgt_1,189,190,232).  visible(fcgt_1,232). happensAt(walking(fcgt_1),232). orientation(fcgt_4,166,232). coords(fcgt_4,80,205,232).  visible(fcgt_4,232). happensAt(inactive(fcgt_4),232).
orientation(fcgt_1,156,233). coords(fcgt_1,188,188,233).  visible(fcgt_1,233). happensAt(walking(fcgt_1),233). orientation(fcgt_4,166,233). coords(fcgt_4,80,205,233).  visible(fcgt_4,233). happensAt(inactive(fcgt_4),233).
orientation(fcgt_1,156,234). coords(fcgt_1,186,188,234).  visible(fcgt_1,234). happensAt(walking(fcgt_1),234). orientation(fcgt_4,166,234). coords(fcgt_4,80,205,234).  visible(fcgt_4,234). happensAt(inactive(fcgt_4),234).
orientation(fcgt_1,149,235). coords(fcgt_1,186,186,235).  visible(fcgt_1,235). happensAt(walking(fcgt_1),235). orientation(fcgt_4,166,235). coords(fcgt_4,80,205,235).  visible(fcgt_4,235). happensAt(inactive(fcgt_4),235).
orientation(fcgt_1,149,236). coords(fcgt_1,184,183,236).  visible(fcgt_1,236). happensAt(walking(fcgt_1),236). orientation(fcgt_4,166,236). coords(fcgt_4,80,205,236).  visible(fcgt_4,236). happensAt(inactive(fcgt_4),236).
orientation(fcgt_1,149,237). coords(fcgt_1,181,181,237).  visible(fcgt_1,237). happensAt(walking(fcgt_1),237). orientation(fcgt_4,166,237). coords(fcgt_4,80,205,237).  visible(fcgt_4,237). happensAt(inactive(fcgt_4),237).
orientation(fcgt_1,149,238). coords(fcgt_1,177,178,238).  visible(fcgt_1,238). happensAt(walking(fcgt_1),238). orientation(fcgt_4,166,238). coords(fcgt_4,80,205,238).  visible(fcgt_4,238). happensAt(inactive(fcgt_4),238).
orientation(fcgt_1,149,239). coords(fcgt_1,175,175,239).  visible(fcgt_1,239). happensAt(walking(fcgt_1),239). orientation(fcgt_4,166,239). coords(fcgt_4,80,205,239).  visible(fcgt_4,239). happensAt(inactive(fcgt_4),239).
orientation(fcgt_1,149,240). coords(fcgt_1,173,171,240).  visible(fcgt_1,240). happensAt(walking(fcgt_1),240). orientation(fcgt_4,166,240). coords(fcgt_4,80,205,240).  visible(fcgt_4,240). happensAt(inactive(fcgt_4),240).
orientation(fcgt_1,149,241). coords(fcgt_1,173,170,241).  visible(fcgt_1,241). happensAt(walking(fcgt_1),241). orientation(fcgt_4,166,241). coords(fcgt_4,80,205,241).  visible(fcgt_4,241). happensAt(inactive(fcgt_4),241).
orientation(fcgt_1,149,242). coords(fcgt_1,172,165,242).  visible(fcgt_1,242). happensAt(walking(fcgt_1),242). orientation(fcgt_4,166,242). coords(fcgt_4,80,205,242).  visible(fcgt_4,242). happensAt(inactive(fcgt_4),242).
orientation(fcgt_1,149,243). coords(fcgt_1,171,163,243).  visible(fcgt_1,243). happensAt(walking(fcgt_1),243). orientation(fcgt_4,166,243). coords(fcgt_4,80,205,243).  visible(fcgt_4,243). happensAt(inactive(fcgt_4),243).
orientation(fcgt_1,154,244). coords(fcgt_1,171,162,244).  visible(fcgt_1,244). happensAt(walking(fcgt_1),244). orientation(fcgt_4,166,244). coords(fcgt_4,80,205,244).  visible(fcgt_4,244). happensAt(inactive(fcgt_4),244).
#show holdsAt/2.