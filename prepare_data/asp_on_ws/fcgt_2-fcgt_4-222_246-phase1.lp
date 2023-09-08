
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

	orientation(fcgt_2,157,222). coords(fcgt_2,48,178,222).  visible(fcgt_2,222). happensAt(walking(fcgt_2),222). orientation(fcgt_4,172,222). coords(fcgt_4,81,205,222).  visible(fcgt_4,222). happensAt(walking(fcgt_4),222).
orientation(fcgt_2,157,223). coords(fcgt_2,48,177,223).  visible(fcgt_2,223). happensAt(walking(fcgt_2),223). orientation(fcgt_4,166,223). coords(fcgt_4,81,205,223).  visible(fcgt_4,223). happensAt(inactive(fcgt_4),223).
orientation(fcgt_2,157,224). coords(fcgt_2,48,176,224).  visible(fcgt_2,224). happensAt(walking(fcgt_2),224). orientation(fcgt_4,166,224). coords(fcgt_4,81,205,224).  visible(fcgt_4,224). happensAt(inactive(fcgt_4),224).
orientation(fcgt_2,157,225). coords(fcgt_2,49,176,225).  visible(fcgt_2,225). happensAt(walking(fcgt_2),225). orientation(fcgt_4,166,225). coords(fcgt_4,80,205,225).  visible(fcgt_4,225). happensAt(inactive(fcgt_4),225).
orientation(fcgt_2,157,226). coords(fcgt_2,50,176,226).  visible(fcgt_2,226). happensAt(walking(fcgt_2),226). orientation(fcgt_4,166,226). coords(fcgt_4,80,205,226).  visible(fcgt_4,226). happensAt(inactive(fcgt_4),226).
orientation(fcgt_2,157,227). coords(fcgt_2,51,175,227).  visible(fcgt_2,227). happensAt(walking(fcgt_2),227). orientation(fcgt_4,166,227). coords(fcgt_4,80,205,227).  visible(fcgt_4,227). happensAt(inactive(fcgt_4),227).
orientation(fcgt_2,157,228). coords(fcgt_2,51,175,228).  visible(fcgt_2,228). happensAt(walking(fcgt_2),228). orientation(fcgt_4,166,228). coords(fcgt_4,80,205,228).  visible(fcgt_4,228). happensAt(inactive(fcgt_4),228).
orientation(fcgt_2,157,229). coords(fcgt_2,51,175,229).  visible(fcgt_2,229). happensAt(walking(fcgt_2),229). orientation(fcgt_4,166,229). coords(fcgt_4,80,205,229).  visible(fcgt_4,229). happensAt(inactive(fcgt_4),229).
orientation(fcgt_2,157,230). coords(fcgt_2,52,174,230).  visible(fcgt_2,230). happensAt(walking(fcgt_2),230). orientation(fcgt_4,166,230). coords(fcgt_4,80,205,230).  visible(fcgt_4,230). happensAt(inactive(fcgt_4),230).
orientation(fcgt_2,157,231). coords(fcgt_2,53,174,231).  visible(fcgt_2,231). happensAt(walking(fcgt_2),231). orientation(fcgt_4,166,231). coords(fcgt_4,80,205,231).  visible(fcgt_4,231). happensAt(inactive(fcgt_4),231).
orientation(fcgt_2,157,232). coords(fcgt_2,53,172,232).  visible(fcgt_2,232). happensAt(walking(fcgt_2),232). orientation(fcgt_4,166,232). coords(fcgt_4,80,205,232).  visible(fcgt_4,232). happensAt(inactive(fcgt_4),232).
orientation(fcgt_2,157,233). coords(fcgt_2,53,171,233).  visible(fcgt_2,233). happensAt(walking(fcgt_2),233). orientation(fcgt_4,166,233). coords(fcgt_4,80,205,233).  visible(fcgt_4,233). happensAt(inactive(fcgt_4),233).
orientation(fcgt_2,157,234). coords(fcgt_2,53,170,234).  visible(fcgt_2,234). happensAt(walking(fcgt_2),234). orientation(fcgt_4,166,234). coords(fcgt_4,80,205,234).  visible(fcgt_4,234). happensAt(inactive(fcgt_4),234).
orientation(fcgt_2,157,235). coords(fcgt_2,53,169,235).  visible(fcgt_2,235). happensAt(walking(fcgt_2),235). orientation(fcgt_4,166,235). coords(fcgt_4,80,205,235).  visible(fcgt_4,235). happensAt(inactive(fcgt_4),235).
orientation(fcgt_2,157,236). coords(fcgt_2,53,169,236).  visible(fcgt_2,236). happensAt(walking(fcgt_2),236). orientation(fcgt_4,166,236). coords(fcgt_4,80,205,236).  visible(fcgt_4,236). happensAt(inactive(fcgt_4),236).
orientation(fcgt_2,157,237). coords(fcgt_2,55,168,237).  visible(fcgt_2,237). happensAt(walking(fcgt_2),237). orientation(fcgt_4,166,237). coords(fcgt_4,80,205,237).  visible(fcgt_4,237). happensAt(inactive(fcgt_4),237).
orientation(fcgt_2,157,238). coords(fcgt_2,56,167,238).  visible(fcgt_2,238). happensAt(walking(fcgt_2),238). orientation(fcgt_4,166,238). coords(fcgt_4,80,205,238).  visible(fcgt_4,238). happensAt(inactive(fcgt_4),238).
orientation(fcgt_2,157,239). coords(fcgt_2,56,167,239).  visible(fcgt_2,239). happensAt(walking(fcgt_2),239). orientation(fcgt_4,166,239). coords(fcgt_4,80,205,239).  visible(fcgt_4,239). happensAt(inactive(fcgt_4),239).
orientation(fcgt_2,157,240). coords(fcgt_2,56,166,240).  visible(fcgt_2,240). happensAt(walking(fcgt_2),240). orientation(fcgt_4,166,240). coords(fcgt_4,80,205,240).  visible(fcgt_4,240). happensAt(inactive(fcgt_4),240).
orientation(fcgt_2,157,241). coords(fcgt_2,57,165,241).  visible(fcgt_2,241). happensAt(walking(fcgt_2),241). orientation(fcgt_4,166,241). coords(fcgt_4,80,205,241).  visible(fcgt_4,241). happensAt(inactive(fcgt_4),241).
orientation(fcgt_2,157,242). coords(fcgt_2,57,165,242).  visible(fcgt_2,242). happensAt(walking(fcgt_2),242). orientation(fcgt_4,166,242). coords(fcgt_4,80,205,242).  visible(fcgt_4,242). happensAt(inactive(fcgt_4),242).
orientation(fcgt_2,157,243). coords(fcgt_2,57,163,243).  visible(fcgt_2,243). happensAt(walking(fcgt_2),243). orientation(fcgt_4,166,243). coords(fcgt_4,80,205,243).  visible(fcgt_4,243). happensAt(inactive(fcgt_4),243).
orientation(fcgt_2,157,244). coords(fcgt_2,58,163,244).  visible(fcgt_2,244). happensAt(walking(fcgt_2),244). orientation(fcgt_4,166,244). coords(fcgt_4,80,205,244).  visible(fcgt_4,244). happensAt(inactive(fcgt_4),244).
orientation(fcgt_2,157,245). coords(fcgt_2,58,162,245).  visible(fcgt_2,245). happensAt(walking(fcgt_2),245). orientation(fcgt_4,166,245). coords(fcgt_4,80,205,245).  visible(fcgt_4,245). happensAt(inactive(fcgt_4),245).
orientation(fcgt_2,157,246). coords(fcgt_2,58,161,246).  visible(fcgt_2,246). happensAt(walking(fcgt_2),246). orientation(fcgt_4,166,246). coords(fcgt_4,80,205,246).  visible(fcgt_4,246). happensAt(inactive(fcgt_4),246).
#show holdsAt/2.