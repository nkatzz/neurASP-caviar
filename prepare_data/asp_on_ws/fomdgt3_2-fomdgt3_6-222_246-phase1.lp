
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

	orientation(fomdgt3_2,130,222). coords(fomdgt3_2,228,128,222).  visible(fomdgt3_2,222). happensAt(walking(fomdgt3_2),222). orientation(fomdgt3_6,130,222). coords(fomdgt3_6,205,89,222).  visible(fomdgt3_6,222). happensAt(walking(fomdgt3_6),222).
orientation(fomdgt3_2,130,223). coords(fomdgt3_2,228,128,223).  visible(fomdgt3_2,223). happensAt(walking(fomdgt3_2),223). orientation(fomdgt3_6,130,223). coords(fomdgt3_6,205,89,223).  visible(fomdgt3_6,223). happensAt(walking(fomdgt3_6),223).
orientation(fomdgt3_2,130,224). coords(fomdgt3_2,228,128,224).  visible(fomdgt3_2,224). happensAt(walking(fomdgt3_2),224). orientation(fomdgt3_6,130,224). coords(fomdgt3_6,205,90,224).  visible(fomdgt3_6,224). happensAt(walking(fomdgt3_6),224).
orientation(fomdgt3_2,130,225). coords(fomdgt3_2,228,128,225).  visible(fomdgt3_2,225). happensAt(walking(fomdgt3_2),225). orientation(fomdgt3_6,130,225). coords(fomdgt3_6,205,90,225).  visible(fomdgt3_6,225). happensAt(walking(fomdgt3_6),225).
orientation(fomdgt3_2,130,226). coords(fomdgt3_2,228,128,226).  visible(fomdgt3_2,226). happensAt(walking(fomdgt3_2),226). orientation(fomdgt3_6,130,226). coords(fomdgt3_6,205,90,226).  visible(fomdgt3_6,226). happensAt(walking(fomdgt3_6),226).
orientation(fomdgt3_2,130,227). coords(fomdgt3_2,228,128,227).  visible(fomdgt3_2,227). happensAt(walking(fomdgt3_2),227). orientation(fomdgt3_6,130,227). coords(fomdgt3_6,205,92,227).  visible(fomdgt3_6,227). happensAt(walking(fomdgt3_6),227).
orientation(fomdgt3_2,130,228). coords(fomdgt3_2,228,128,228).  visible(fomdgt3_2,228). happensAt(walking(fomdgt3_2),228). orientation(fomdgt3_6,130,228). coords(fomdgt3_6,205,92,228).  visible(fomdgt3_6,228). happensAt(walking(fomdgt3_6),228).
orientation(fomdgt3_2,130,229). coords(fomdgt3_2,228,128,229).  visible(fomdgt3_2,229). happensAt(walking(fomdgt3_2),229). orientation(fomdgt3_6,130,229). coords(fomdgt3_6,205,95,229).  visible(fomdgt3_6,229). happensAt(walking(fomdgt3_6),229).
orientation(fomdgt3_2,130,230). coords(fomdgt3_2,228,128,230).  visible(fomdgt3_2,230). happensAt(walking(fomdgt3_2),230). orientation(fomdgt3_6,130,230). coords(fomdgt3_6,216,97,230).  visible(fomdgt3_6,230). happensAt(walking(fomdgt3_6),230).
orientation(fomdgt3_2,130,231). coords(fomdgt3_2,228,128,231).  visible(fomdgt3_2,231). happensAt(walking(fomdgt3_2),231). orientation(fomdgt3_6,130,231). coords(fomdgt3_6,216,97,231).  visible(fomdgt3_6,231). happensAt(walking(fomdgt3_6),231).
orientation(fomdgt3_2,130,232). coords(fomdgt3_2,228,128,232).  visible(fomdgt3_2,232). happensAt(walking(fomdgt3_2),232). orientation(fomdgt3_6,130,232). coords(fomdgt3_6,216,97,232).  visible(fomdgt3_6,232). happensAt(walking(fomdgt3_6),232).
orientation(fomdgt3_2,130,233). coords(fomdgt3_2,228,128,233).  visible(fomdgt3_2,233). happensAt(walking(fomdgt3_2),233). orientation(fomdgt3_6,130,233). coords(fomdgt3_6,216,97,233).  visible(fomdgt3_6,233). happensAt(walking(fomdgt3_6),233).
orientation(fomdgt3_2,130,234). coords(fomdgt3_2,228,128,234).  visible(fomdgt3_2,234). happensAt(walking(fomdgt3_2),234). orientation(fomdgt3_6,130,234). coords(fomdgt3_6,216,104,234).  visible(fomdgt3_6,234). happensAt(walking(fomdgt3_6),234).
orientation(fomdgt3_2,130,235). coords(fomdgt3_2,228,128,235).  visible(fomdgt3_2,235). happensAt(walking(fomdgt3_2),235). orientation(fomdgt3_6,130,235). coords(fomdgt3_6,216,104,235).  visible(fomdgt3_6,235). happensAt(walking(fomdgt3_6),235).
orientation(fomdgt3_2,130,236). coords(fomdgt3_2,228,128,236).  visible(fomdgt3_2,236). happensAt(walking(fomdgt3_2),236). orientation(fomdgt3_6,130,236). coords(fomdgt3_6,212,104,236).  visible(fomdgt3_6,236). happensAt(walking(fomdgt3_6),236).
orientation(fomdgt3_2,130,237). coords(fomdgt3_2,228,128,237).  visible(fomdgt3_2,237). happensAt(walking(fomdgt3_2),237). orientation(fomdgt3_6,130,237). coords(fomdgt3_6,209,106,237).  visible(fomdgt3_6,237). happensAt(walking(fomdgt3_6),237).
orientation(fomdgt3_2,130,238). coords(fomdgt3_2,228,128,238).  visible(fomdgt3_2,238). happensAt(walking(fomdgt3_2),238). orientation(fomdgt3_6,130,238). coords(fomdgt3_6,209,104,238).  visible(fomdgt3_6,238). happensAt(walking(fomdgt3_6),238).
orientation(fomdgt3_2,130,239). coords(fomdgt3_2,228,128,239).  visible(fomdgt3_2,239). happensAt(walking(fomdgt3_2),239). orientation(fomdgt3_6,130,239). coords(fomdgt3_6,209,104,239).  visible(fomdgt3_6,239). happensAt(walking(fomdgt3_6),239).
orientation(fomdgt3_2,130,240). coords(fomdgt3_2,228,128,240).  visible(fomdgt3_2,240). happensAt(walking(fomdgt3_2),240). orientation(fomdgt3_6,130,240). coords(fomdgt3_6,209,111,240).  visible(fomdgt3_6,240). happensAt(walking(fomdgt3_6),240).
orientation(fomdgt3_2,130,241). coords(fomdgt3_2,228,128,241).  visible(fomdgt3_2,241). happensAt(walking(fomdgt3_2),241). orientation(fomdgt3_6,130,241). coords(fomdgt3_6,213,114,241).  visible(fomdgt3_6,241). happensAt(walking(fomdgt3_6),241).
orientation(fomdgt3_2,130,242). coords(fomdgt3_2,228,128,242).  visible(fomdgt3_2,242). happensAt(walking(fomdgt3_2),242). orientation(fomdgt3_6,130,242). coords(fomdgt3_6,215,120,242).  visible(fomdgt3_6,242). happensAt(walking(fomdgt3_6),242).
orientation(fomdgt3_2,130,243). coords(fomdgt3_2,228,128,243).  visible(fomdgt3_2,243). happensAt(walking(fomdgt3_2),243). orientation(fomdgt3_6,130,243). coords(fomdgt3_6,215,122,243).  visible(fomdgt3_6,243). happensAt(walking(fomdgt3_6),243).
orientation(fomdgt3_2,130,244). coords(fomdgt3_2,228,128,244).  visible(fomdgt3_2,244). happensAt(walking(fomdgt3_2),244). orientation(fomdgt3_6,130,244). coords(fomdgt3_6,210,129,244).  visible(fomdgt3_6,244). happensAt(walking(fomdgt3_6),244).
orientation(fomdgt3_2,130,245). coords(fomdgt3_2,228,128,245).  visible(fomdgt3_2,245). happensAt(walking(fomdgt3_2),245). orientation(fomdgt3_6,130,245). coords(fomdgt3_6,210,129,245).  visible(fomdgt3_6,245). happensAt(walking(fomdgt3_6),245).
orientation(fomdgt3_2,130,246). coords(fomdgt3_2,228,128,246).  visible(fomdgt3_2,246). happensAt(walking(fomdgt3_2),246). orientation(fomdgt3_6,130,246). coords(fomdgt3_6,209,133,246).  visible(fomdgt3_6,246). happensAt(walking(fomdgt3_6),246).
#show holdsAt/2.