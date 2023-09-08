
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

	orientation(fcgt_2,157,222). coords(fcgt_2,48,178,222).  visible(fcgt_2,222). happensAt(walking(fcgt_2),222). orientation(fcgt_3,149,222). coords(fcgt_3,100,170,222).  visible(fcgt_3,222). happensAt(walking(fcgt_3),222).
orientation(fcgt_2,157,223). coords(fcgt_2,48,177,223).  visible(fcgt_2,223). happensAt(walking(fcgt_2),223). orientation(fcgt_3,149,223). coords(fcgt_3,100,170,223).  visible(fcgt_3,223). happensAt(walking(fcgt_3),223).
orientation(fcgt_2,157,224). coords(fcgt_2,48,176,224).  visible(fcgt_2,224). happensAt(walking(fcgt_2),224). orientation(fcgt_3,149,224). coords(fcgt_3,100,170,224).  visible(fcgt_3,224). happensAt(walking(fcgt_3),224).
orientation(fcgt_2,157,225). coords(fcgt_2,49,176,225).  visible(fcgt_2,225). happensAt(walking(fcgt_2),225). orientation(fcgt_3,149,225). coords(fcgt_3,100,171,225).  visible(fcgt_3,225). happensAt(walking(fcgt_3),225).
orientation(fcgt_2,157,226). coords(fcgt_2,50,176,226).  visible(fcgt_2,226). happensAt(walking(fcgt_2),226). orientation(fcgt_3,149,226). coords(fcgt_3,100,171,226).  visible(fcgt_3,226). happensAt(walking(fcgt_3),226).
orientation(fcgt_2,157,227). coords(fcgt_2,51,175,227).  visible(fcgt_2,227). happensAt(walking(fcgt_2),227). orientation(fcgt_3,157,227). coords(fcgt_3,100,171,227).  visible(fcgt_3,227). happensAt(walking(fcgt_3),227).
orientation(fcgt_2,157,228). coords(fcgt_2,51,175,228).  visible(fcgt_2,228). happensAt(walking(fcgt_2),228). orientation(fcgt_3,159,228). coords(fcgt_3,100,171,228).  visible(fcgt_3,228). happensAt(walking(fcgt_3),228).
orientation(fcgt_2,157,229). coords(fcgt_2,51,175,229).  visible(fcgt_2,229). happensAt(walking(fcgt_2),229). orientation(fcgt_3,159,229). coords(fcgt_3,100,172,229).  visible(fcgt_3,229). happensAt(walking(fcgt_3),229).
orientation(fcgt_2,157,230). coords(fcgt_2,52,174,230).  visible(fcgt_2,230). happensAt(walking(fcgt_2),230). orientation(fcgt_3,159,230). coords(fcgt_3,100,172,230).  visible(fcgt_3,230). happensAt(walking(fcgt_3),230).
orientation(fcgt_2,157,231). coords(fcgt_2,53,174,231).  visible(fcgt_2,231). happensAt(walking(fcgt_2),231). orientation(fcgt_3,159,231). coords(fcgt_3,100,172,231).  visible(fcgt_3,231). happensAt(walking(fcgt_3),231).
orientation(fcgt_2,157,232). coords(fcgt_2,53,172,232).  visible(fcgt_2,232). happensAt(walking(fcgt_2),232). orientation(fcgt_3,159,232). coords(fcgt_3,100,172,232).  visible(fcgt_3,232). happensAt(walking(fcgt_3),232).
orientation(fcgt_2,157,233). coords(fcgt_2,53,171,233).  visible(fcgt_2,233). happensAt(walking(fcgt_2),233). orientation(fcgt_3,159,233). coords(fcgt_3,100,172,233).  visible(fcgt_3,233). happensAt(walking(fcgt_3),233).
orientation(fcgt_2,157,234). coords(fcgt_2,53,170,234).  visible(fcgt_2,234). happensAt(walking(fcgt_2),234). orientation(fcgt_3,159,234). coords(fcgt_3,100,172,234).  visible(fcgt_3,234). happensAt(walking(fcgt_3),234).
orientation(fcgt_2,157,235). coords(fcgt_2,53,169,235).  visible(fcgt_2,235). happensAt(walking(fcgt_2),235). orientation(fcgt_3,159,235). coords(fcgt_3,100,173,235).  visible(fcgt_3,235). happensAt(walking(fcgt_3),235).
orientation(fcgt_2,157,236). coords(fcgt_2,53,169,236).  visible(fcgt_2,236). happensAt(walking(fcgt_2),236). orientation(fcgt_3,159,236). coords(fcgt_3,100,173,236).  visible(fcgt_3,236). happensAt(walking(fcgt_3),236).
orientation(fcgt_2,157,237). coords(fcgt_2,55,168,237).  visible(fcgt_2,237). happensAt(walking(fcgt_2),237). orientation(fcgt_3,159,237). coords(fcgt_3,100,173,237).  visible(fcgt_3,237). happensAt(walking(fcgt_3),237).
orientation(fcgt_2,157,238). coords(fcgt_2,56,167,238).  visible(fcgt_2,238). happensAt(walking(fcgt_2),238). orientation(fcgt_3,159,238). coords(fcgt_3,100,174,238).  visible(fcgt_3,238). happensAt(walking(fcgt_3),238).
orientation(fcgt_2,157,239). coords(fcgt_2,56,167,239).  visible(fcgt_2,239). happensAt(walking(fcgt_2),239). orientation(fcgt_3,163,239). coords(fcgt_3,101,174,239).  visible(fcgt_3,239). happensAt(walking(fcgt_3),239).
orientation(fcgt_2,157,240). coords(fcgt_2,56,166,240).  visible(fcgt_2,240). happensAt(walking(fcgt_2),240). orientation(fcgt_3,166,240). coords(fcgt_3,101,175,240).  visible(fcgt_3,240). happensAt(walking(fcgt_3),240).
orientation(fcgt_2,157,241). coords(fcgt_2,57,165,241).  visible(fcgt_2,241). happensAt(walking(fcgt_2),241). orientation(fcgt_3,167,241). coords(fcgt_3,101,175,241).  visible(fcgt_3,241). happensAt(walking(fcgt_3),241).
orientation(fcgt_2,157,242). coords(fcgt_2,57,165,242).  visible(fcgt_2,242). happensAt(walking(fcgt_2),242). orientation(fcgt_3,166,242). coords(fcgt_3,101,175,242).  visible(fcgt_3,242). happensAt(walking(fcgt_3),242).
orientation(fcgt_2,157,243). coords(fcgt_2,57,163,243).  visible(fcgt_2,243). happensAt(walking(fcgt_2),243). orientation(fcgt_3,167,243). coords(fcgt_3,101,175,243).  visible(fcgt_3,243). happensAt(walking(fcgt_3),243).
orientation(fcgt_2,157,244). coords(fcgt_2,58,163,244).  visible(fcgt_2,244). happensAt(walking(fcgt_2),244). orientation(fcgt_3,170,244). coords(fcgt_3,101,175,244).  visible(fcgt_3,244). happensAt(walking(fcgt_3),244).
orientation(fcgt_2,157,245). coords(fcgt_2,58,162,245).  visible(fcgt_2,245). happensAt(walking(fcgt_2),245). orientation(fcgt_3,170,245). coords(fcgt_3,101,175,245).  visible(fcgt_3,245). happensAt(walking(fcgt_3),245).
orientation(fcgt_2,157,246). coords(fcgt_2,58,161,246).  visible(fcgt_2,246). happensAt(walking(fcgt_2),246). orientation(fcgt_3,170,246). coords(fcgt_3,101,176,246).  visible(fcgt_3,246). happensAt(walking(fcgt_3),246).
#show holdsAt/2.