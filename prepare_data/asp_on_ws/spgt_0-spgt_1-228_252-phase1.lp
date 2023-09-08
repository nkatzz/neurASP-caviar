
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

	orientation(spgt_0,148,228). coords(spgt_0,113,114,228).  visible(spgt_0,228). happensAt(inactive(spgt_0),228). orientation(spgt_1,148,228). coords(spgt_1,109,101,228).  visible(spgt_1,228). happensAt(active(spgt_1),228).
orientation(spgt_0,148,229). coords(spgt_0,113,114,229).  visible(spgt_0,229). happensAt(inactive(spgt_0),229). orientation(spgt_1,148,229). coords(spgt_1,109,101,229).  visible(spgt_1,229). happensAt(active(spgt_1),229).
orientation(spgt_0,148,230). coords(spgt_0,113,114,230).  visible(spgt_0,230). happensAt(inactive(spgt_0),230). orientation(spgt_1,148,230). coords(spgt_1,109,100,230).  visible(spgt_1,230). happensAt(active(spgt_1),230).
orientation(spgt_0,148,231). coords(spgt_0,113,114,231).  visible(spgt_0,231). happensAt(inactive(spgt_0),231). orientation(spgt_1,148,231). coords(spgt_1,110,100,231).  visible(spgt_1,231). happensAt(active(spgt_1),231).
orientation(spgt_0,148,232). coords(spgt_0,113,114,232).  visible(spgt_0,232). happensAt(inactive(spgt_0),232). orientation(spgt_1,147,232). coords(spgt_1,110,100,232).  visible(spgt_1,232). happensAt(active(spgt_1),232).
orientation(spgt_0,148,233). coords(spgt_0,114,114,233).  visible(spgt_0,233). happensAt(active(spgt_0),233). orientation(spgt_1,144,233). coords(spgt_1,110,99,233).  visible(spgt_1,233). happensAt(active(spgt_1),233).
orientation(spgt_0,148,234). coords(spgt_0,114,114,234).  visible(spgt_0,234). happensAt(active(spgt_0),234). orientation(spgt_1,141,234). coords(spgt_1,110,99,234).  visible(spgt_1,234). happensAt(active(spgt_1),234).
orientation(spgt_0,148,235). coords(spgt_0,114,114,235).  visible(spgt_0,235). happensAt(active(spgt_0),235). orientation(spgt_1,141,235). coords(spgt_1,110,99,235).  visible(spgt_1,235). happensAt(active(spgt_1),235).
orientation(spgt_0,148,236). coords(spgt_0,115,113,236).  visible(spgt_0,236). happensAt(active(spgt_0),236). orientation(spgt_1,141,236). coords(spgt_1,111,98,236).  visible(spgt_1,236). happensAt(walking(spgt_1),236).
orientation(spgt_0,148,237). coords(spgt_0,115,113,237).  visible(spgt_0,237). happensAt(active(spgt_0),237). orientation(spgt_1,141,237). coords(spgt_1,112,98,237).  visible(spgt_1,237). happensAt(walking(spgt_1),237).
orientation(spgt_0,148,238). coords(spgt_0,116,113,238).  visible(spgt_0,238). happensAt(active(spgt_0),238). orientation(spgt_1,141,238). coords(spgt_1,113,98,238).  visible(spgt_1,238). happensAt(walking(spgt_1),238).
orientation(spgt_0,148,239). coords(spgt_0,116,113,239).  visible(spgt_0,239). happensAt(active(spgt_0),239). orientation(spgt_1,141,239). coords(spgt_1,114,97,239).  visible(spgt_1,239). happensAt(walking(spgt_1),239).
orientation(spgt_0,148,240). coords(spgt_0,116,112,240).  visible(spgt_0,240). happensAt(active(spgt_0),240). orientation(spgt_1,141,240). coords(spgt_1,115,96,240).  visible(spgt_1,240). happensAt(walking(spgt_1),240).
orientation(spgt_0,148,241). coords(spgt_0,117,113,241).  visible(spgt_0,241). happensAt(active(spgt_0),241). orientation(spgt_1,141,241). coords(spgt_1,116,96,241).  visible(spgt_1,241). happensAt(walking(spgt_1),241).
orientation(spgt_0,148,242). coords(spgt_0,118,112,242).  visible(spgt_0,242). happensAt(active(spgt_0),242). orientation(spgt_1,141,242). coords(spgt_1,116,96,242).  visible(spgt_1,242). happensAt(walking(spgt_1),242).
orientation(spgt_0,148,243). coords(spgt_0,118,112,243).  visible(spgt_0,243). happensAt(active(spgt_0),243). orientation(spgt_1,141,243). coords(spgt_1,116,95,243).  visible(spgt_1,243). happensAt(walking(spgt_1),243).
orientation(spgt_0,148,244). coords(spgt_0,118,112,244).  visible(spgt_0,244). happensAt(active(spgt_0),244). orientation(spgt_1,141,244). coords(spgt_1,116,95,244).  visible(spgt_1,244). happensAt(walking(spgt_1),244).
orientation(spgt_0,148,245). coords(spgt_0,118,111,245).  visible(spgt_0,245). happensAt(active(spgt_0),245). orientation(spgt_1,137,245). coords(spgt_1,117,94,245).  visible(spgt_1,245). happensAt(walking(spgt_1),245).
orientation(spgt_0,148,246). coords(spgt_0,119,111,246).  visible(spgt_0,246). happensAt(active(spgt_0),246). orientation(spgt_1,137,246). coords(spgt_1,117,93,246).  visible(spgt_1,246). happensAt(walking(spgt_1),246).
orientation(spgt_0,148,247). coords(spgt_0,119,112,247).  visible(spgt_0,247). happensAt(active(spgt_0),247). orientation(spgt_1,137,247). coords(spgt_1,117,91,247).  visible(spgt_1,247). happensAt(walking(spgt_1),247).
orientation(spgt_0,148,248). coords(spgt_0,119,111,248).  visible(spgt_0,248). happensAt(active(spgt_0),248). orientation(spgt_1,137,248). coords(spgt_1,117,90,248).  visible(spgt_1,248). happensAt(walking(spgt_1),248).
orientation(spgt_0,148,249). coords(spgt_0,119,111,249).  visible(spgt_0,249). happensAt(active(spgt_0),249). orientation(spgt_1,137,249). coords(spgt_1,118,89,249).  visible(spgt_1,249). happensAt(walking(spgt_1),249).
orientation(spgt_0,148,250). coords(spgt_0,120,111,250).  visible(spgt_0,250). happensAt(walking(spgt_0),250). orientation(spgt_1,137,250). coords(spgt_1,118,87,250).  visible(spgt_1,250). happensAt(walking(spgt_1),250).
orientation(spgt_0,148,251). coords(spgt_0,120,110,251).  visible(spgt_0,251). happensAt(walking(spgt_0),251). orientation(spgt_1,137,251). coords(spgt_1,118,87,251).  visible(spgt_1,251). happensAt(walking(spgt_1),251).
orientation(spgt_0,148,252). coords(spgt_0,123,110,252).  visible(spgt_0,252). happensAt(walking(spgt_0),252). orientation(spgt_1,137,252). coords(spgt_1,119,86,252).  visible(spgt_1,252). happensAt(walking(spgt_1),252).
#show holdsAt/2.