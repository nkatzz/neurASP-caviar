
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

	orientation(fcgt_1,171,220). coords(fcgt_1,210,212,220).  visible(fcgt_1,220). happensAt(walking(fcgt_1),220). orientation(fcgt_2,157,220). coords(fcgt_2,48,180,220).  visible(fcgt_2,220). happensAt(walking(fcgt_2),220).
orientation(fcgt_1,171,221). coords(fcgt_1,210,210,221).  visible(fcgt_1,221). happensAt(walking(fcgt_1),221). orientation(fcgt_2,157,221). coords(fcgt_2,48,179,221).  visible(fcgt_2,221). happensAt(walking(fcgt_2),221).
orientation(fcgt_1,171,222). coords(fcgt_1,209,209,222).  visible(fcgt_1,222). happensAt(walking(fcgt_1),222). orientation(fcgt_2,157,222). coords(fcgt_2,48,178,222).  visible(fcgt_2,222). happensAt(walking(fcgt_2),222).
orientation(fcgt_1,171,223). coords(fcgt_1,206,206,223).  visible(fcgt_1,223). happensAt(walking(fcgt_1),223). orientation(fcgt_2,157,223). coords(fcgt_2,48,177,223).  visible(fcgt_2,223). happensAt(walking(fcgt_2),223).
orientation(fcgt_1,171,224). coords(fcgt_1,205,204,224).  visible(fcgt_1,224). happensAt(walking(fcgt_1),224). orientation(fcgt_2,157,224). coords(fcgt_2,48,176,224).  visible(fcgt_2,224). happensAt(walking(fcgt_2),224).
orientation(fcgt_1,171,225). coords(fcgt_1,203,202,225).  visible(fcgt_1,225). happensAt(walking(fcgt_1),225). orientation(fcgt_2,157,225). coords(fcgt_2,49,176,225).  visible(fcgt_2,225). happensAt(walking(fcgt_2),225).
orientation(fcgt_1,171,226). coords(fcgt_1,200,199,226).  visible(fcgt_1,226). happensAt(walking(fcgt_1),226). orientation(fcgt_2,157,226). coords(fcgt_2,50,176,226).  visible(fcgt_2,226). happensAt(walking(fcgt_2),226).
orientation(fcgt_1,171,227). coords(fcgt_1,197,197,227).  visible(fcgt_1,227). happensAt(walking(fcgt_1),227). orientation(fcgt_2,157,227). coords(fcgt_2,51,175,227).  visible(fcgt_2,227). happensAt(walking(fcgt_2),227).
orientation(fcgt_1,171,228). coords(fcgt_1,194,194,228).  visible(fcgt_1,228). happensAt(walking(fcgt_1),228). orientation(fcgt_2,157,228). coords(fcgt_2,51,175,228).  visible(fcgt_2,228). happensAt(walking(fcgt_2),228).
orientation(fcgt_1,167,229). coords(fcgt_1,191,194,229).  visible(fcgt_1,229). happensAt(walking(fcgt_1),229). orientation(fcgt_2,157,229). coords(fcgt_2,51,175,229).  visible(fcgt_2,229). happensAt(walking(fcgt_2),229).
orientation(fcgt_1,163,230). coords(fcgt_1,190,193,230).  visible(fcgt_1,230). happensAt(walking(fcgt_1),230). orientation(fcgt_2,157,230). coords(fcgt_2,52,174,230).  visible(fcgt_2,230). happensAt(walking(fcgt_2),230).
orientation(fcgt_1,163,231). coords(fcgt_1,189,192,231).  visible(fcgt_1,231). happensAt(walking(fcgt_1),231). orientation(fcgt_2,157,231). coords(fcgt_2,53,174,231).  visible(fcgt_2,231). happensAt(walking(fcgt_2),231).
orientation(fcgt_1,163,232). coords(fcgt_1,189,190,232).  visible(fcgt_1,232). happensAt(walking(fcgt_1),232). orientation(fcgt_2,157,232). coords(fcgt_2,53,172,232).  visible(fcgt_2,232). happensAt(walking(fcgt_2),232).
orientation(fcgt_1,156,233). coords(fcgt_1,188,188,233).  visible(fcgt_1,233). happensAt(walking(fcgt_1),233). orientation(fcgt_2,157,233). coords(fcgt_2,53,171,233).  visible(fcgt_2,233). happensAt(walking(fcgt_2),233).
orientation(fcgt_1,156,234). coords(fcgt_1,186,188,234).  visible(fcgt_1,234). happensAt(walking(fcgt_1),234). orientation(fcgt_2,157,234). coords(fcgt_2,53,170,234).  visible(fcgt_2,234). happensAt(walking(fcgt_2),234).
orientation(fcgt_1,149,235). coords(fcgt_1,186,186,235).  visible(fcgt_1,235). happensAt(walking(fcgt_1),235). orientation(fcgt_2,157,235). coords(fcgt_2,53,169,235).  visible(fcgt_2,235). happensAt(walking(fcgt_2),235).
orientation(fcgt_1,149,236). coords(fcgt_1,184,183,236).  visible(fcgt_1,236). happensAt(walking(fcgt_1),236). orientation(fcgt_2,157,236). coords(fcgt_2,53,169,236).  visible(fcgt_2,236). happensAt(walking(fcgt_2),236).
orientation(fcgt_1,149,237). coords(fcgt_1,181,181,237).  visible(fcgt_1,237). happensAt(walking(fcgt_1),237). orientation(fcgt_2,157,237). coords(fcgt_2,55,168,237).  visible(fcgt_2,237). happensAt(walking(fcgt_2),237).
orientation(fcgt_1,149,238). coords(fcgt_1,177,178,238).  visible(fcgt_1,238). happensAt(walking(fcgt_1),238). orientation(fcgt_2,157,238). coords(fcgt_2,56,167,238).  visible(fcgt_2,238). happensAt(walking(fcgt_2),238).
orientation(fcgt_1,149,239). coords(fcgt_1,175,175,239).  visible(fcgt_1,239). happensAt(walking(fcgt_1),239). orientation(fcgt_2,157,239). coords(fcgt_2,56,167,239).  visible(fcgt_2,239). happensAt(walking(fcgt_2),239).
orientation(fcgt_1,149,240). coords(fcgt_1,173,171,240).  visible(fcgt_1,240). happensAt(walking(fcgt_1),240). orientation(fcgt_2,157,240). coords(fcgt_2,56,166,240).  visible(fcgt_2,240). happensAt(walking(fcgt_2),240).
orientation(fcgt_1,149,241). coords(fcgt_1,173,170,241).  visible(fcgt_1,241). happensAt(walking(fcgt_1),241). orientation(fcgt_2,157,241). coords(fcgt_2,57,165,241).  visible(fcgt_2,241). happensAt(walking(fcgt_2),241).
orientation(fcgt_1,149,242). coords(fcgt_1,172,165,242).  visible(fcgt_1,242). happensAt(walking(fcgt_1),242). orientation(fcgt_2,157,242). coords(fcgt_2,57,165,242).  visible(fcgt_2,242). happensAt(walking(fcgt_2),242).
orientation(fcgt_1,149,243). coords(fcgt_1,171,163,243).  visible(fcgt_1,243). happensAt(walking(fcgt_1),243). orientation(fcgt_2,157,243). coords(fcgt_2,57,163,243).  visible(fcgt_2,243). happensAt(walking(fcgt_2),243).
orientation(fcgt_1,154,244). coords(fcgt_1,171,162,244).  visible(fcgt_1,244). happensAt(walking(fcgt_1),244). orientation(fcgt_2,157,244). coords(fcgt_2,58,163,244).  visible(fcgt_2,244). happensAt(walking(fcgt_2),244).
#show holdsAt/2.