
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

	orientation(wk2gt_1,126,228). coords(wk2gt_1,115,53,228).  visible(wk2gt_1,228). happensAt(inactive(wk2gt_1),228). orientation(wk2gt_2,141,228). coords(wk2gt_2,98,62,228).  visible(wk2gt_2,228). happensAt(inactive(wk2gt_2),228).
orientation(wk2gt_1,126,229). coords(wk2gt_1,115,53,229).  visible(wk2gt_1,229). happensAt(inactive(wk2gt_1),229). orientation(wk2gt_2,141,229). coords(wk2gt_2,98,62,229).  visible(wk2gt_2,229). happensAt(inactive(wk2gt_2),229).
orientation(wk2gt_1,126,230). coords(wk2gt_1,115,53,230).  visible(wk2gt_1,230). happensAt(inactive(wk2gt_1),230). orientation(wk2gt_2,141,230). coords(wk2gt_2,98,62,230).  visible(wk2gt_2,230). happensAt(inactive(wk2gt_2),230).
orientation(wk2gt_1,126,231). coords(wk2gt_1,115,53,231).  visible(wk2gt_1,231). happensAt(inactive(wk2gt_1),231). orientation(wk2gt_2,141,231). coords(wk2gt_2,98,62,231).  visible(wk2gt_2,231). happensAt(inactive(wk2gt_2),231).
orientation(wk2gt_1,126,232). coords(wk2gt_1,115,53,232).  visible(wk2gt_1,232). happensAt(inactive(wk2gt_1),232). orientation(wk2gt_2,141,232). coords(wk2gt_2,98,62,232).  visible(wk2gt_2,232). happensAt(inactive(wk2gt_2),232).
orientation(wk2gt_1,126,233). coords(wk2gt_1,115,53,233).  visible(wk2gt_1,233). happensAt(inactive(wk2gt_1),233). orientation(wk2gt_2,141,233). coords(wk2gt_2,98,62,233).  visible(wk2gt_2,233). happensAt(inactive(wk2gt_2),233).
orientation(wk2gt_1,126,234). coords(wk2gt_1,115,53,234).  visible(wk2gt_1,234). happensAt(inactive(wk2gt_1),234). orientation(wk2gt_2,141,234). coords(wk2gt_2,98,62,234).  visible(wk2gt_2,234). happensAt(inactive(wk2gt_2),234).
orientation(wk2gt_1,126,235). coords(wk2gt_1,115,53,235).  visible(wk2gt_1,235). happensAt(inactive(wk2gt_1),235). orientation(wk2gt_2,141,235). coords(wk2gt_2,98,62,235).  visible(wk2gt_2,235). happensAt(inactive(wk2gt_2),235).
orientation(wk2gt_1,126,236). coords(wk2gt_1,115,53,236).  visible(wk2gt_1,236). happensAt(inactive(wk2gt_1),236). orientation(wk2gt_2,141,236). coords(wk2gt_2,98,62,236).  visible(wk2gt_2,236). happensAt(inactive(wk2gt_2),236).
orientation(wk2gt_1,126,237). coords(wk2gt_1,115,53,237).  visible(wk2gt_1,237). happensAt(inactive(wk2gt_1),237). orientation(wk2gt_2,141,237). coords(wk2gt_2,98,62,237).  visible(wk2gt_2,237). happensAt(inactive(wk2gt_2),237).
orientation(wk2gt_1,126,238). coords(wk2gt_1,115,53,238).  visible(wk2gt_1,238). happensAt(inactive(wk2gt_1),238). orientation(wk2gt_2,141,238). coords(wk2gt_2,98,62,238).  visible(wk2gt_2,238). happensAt(inactive(wk2gt_2),238).
orientation(wk2gt_1,126,239). coords(wk2gt_1,115,53,239).  visible(wk2gt_1,239). happensAt(inactive(wk2gt_1),239). orientation(wk2gt_2,141,239). coords(wk2gt_2,98,62,239).  visible(wk2gt_2,239). happensAt(inactive(wk2gt_2),239).
orientation(wk2gt_1,126,240). coords(wk2gt_1,115,53,240).  visible(wk2gt_1,240). happensAt(inactive(wk2gt_1),240). orientation(wk2gt_2,141,240). coords(wk2gt_2,98,62,240).  visible(wk2gt_2,240). happensAt(inactive(wk2gt_2),240).
orientation(wk2gt_1,126,241). coords(wk2gt_1,115,53,241).  visible(wk2gt_1,241). happensAt(inactive(wk2gt_1),241). orientation(wk2gt_2,141,241). coords(wk2gt_2,98,62,241).  visible(wk2gt_2,241). happensAt(inactive(wk2gt_2),241).
orientation(wk2gt_1,126,242). coords(wk2gt_1,115,53,242).  visible(wk2gt_1,242). happensAt(inactive(wk2gt_1),242). orientation(wk2gt_2,141,242). coords(wk2gt_2,98,62,242).  visible(wk2gt_2,242). happensAt(inactive(wk2gt_2),242).
orientation(wk2gt_1,126,243). coords(wk2gt_1,115,53,243).  visible(wk2gt_1,243). happensAt(inactive(wk2gt_1),243). orientation(wk2gt_2,141,243). coords(wk2gt_2,98,62,243).  visible(wk2gt_2,243). happensAt(inactive(wk2gt_2),243).
orientation(wk2gt_1,126,244). coords(wk2gt_1,115,53,244).  visible(wk2gt_1,244). happensAt(inactive(wk2gt_1),244). orientation(wk2gt_2,141,244). coords(wk2gt_2,98,62,244).  visible(wk2gt_2,244). happensAt(inactive(wk2gt_2),244).
orientation(wk2gt_1,126,245). coords(wk2gt_1,115,53,245).  visible(wk2gt_1,245). happensAt(inactive(wk2gt_1),245). orientation(wk2gt_2,141,245). coords(wk2gt_2,98,62,245).  visible(wk2gt_2,245). happensAt(inactive(wk2gt_2),245).
orientation(wk2gt_1,126,246). coords(wk2gt_1,115,53,246).  visible(wk2gt_1,246). happensAt(inactive(wk2gt_1),246). orientation(wk2gt_2,141,246). coords(wk2gt_2,98,62,246).  visible(wk2gt_2,246). happensAt(inactive(wk2gt_2),246).
orientation(wk2gt_1,126,247). coords(wk2gt_1,115,53,247).  visible(wk2gt_1,247). happensAt(inactive(wk2gt_1),247). orientation(wk2gt_2,141,247). coords(wk2gt_2,98,62,247).  visible(wk2gt_2,247). happensAt(inactive(wk2gt_2),247).
orientation(wk2gt_1,126,248). coords(wk2gt_1,115,53,248).  visible(wk2gt_1,248). happensAt(inactive(wk2gt_1),248). orientation(wk2gt_2,141,248). coords(wk2gt_2,98,62,248).  visible(wk2gt_2,248). happensAt(inactive(wk2gt_2),248).
orientation(wk2gt_1,126,249). coords(wk2gt_1,115,53,249).  visible(wk2gt_1,249). happensAt(inactive(wk2gt_1),249). orientation(wk2gt_2,141,249). coords(wk2gt_2,98,62,249).  visible(wk2gt_2,249). happensAt(inactive(wk2gt_2),249).
orientation(wk2gt_1,125,250). coords(wk2gt_1,115,53,250).  visible(wk2gt_1,250). happensAt(inactive(wk2gt_1),250). orientation(wk2gt_2,141,250). coords(wk2gt_2,98,62,250).  visible(wk2gt_2,250). happensAt(inactive(wk2gt_2),250).
orientation(wk2gt_1,125,251). coords(wk2gt_1,115,53,251).  visible(wk2gt_1,251). happensAt(inactive(wk2gt_1),251). orientation(wk2gt_2,141,251). coords(wk2gt_2,98,62,251).  visible(wk2gt_2,251). happensAt(inactive(wk2gt_2),251).
orientation(wk2gt_1,125,252). coords(wk2gt_1,115,53,252).  visible(wk2gt_1,252). happensAt(inactive(wk2gt_1),252). orientation(wk2gt_2,141,252). coords(wk2gt_2,98,62,252).  visible(wk2gt_2,252). happensAt(inactive(wk2gt_2),252).
#show holdsAt/2.