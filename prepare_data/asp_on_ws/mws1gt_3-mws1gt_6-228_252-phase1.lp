
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

	orientation(mws1gt_3,162,228). coords(mws1gt_3,115,149,228).  visible(mws1gt_3,228). happensAt(walking(mws1gt_3),228). orientation(mws1gt_6,146,228). coords(mws1gt_6,52,101,228).  visible(mws1gt_6,228). happensAt(walking(mws1gt_6),228).
orientation(mws1gt_3,162,229). coords(mws1gt_3,113,149,229).  visible(mws1gt_3,229). happensAt(walking(mws1gt_3),229). orientation(mws1gt_6,146,229). coords(mws1gt_6,52,102,229).  visible(mws1gt_6,229). happensAt(walking(mws1gt_6),229).
orientation(mws1gt_3,162,230). coords(mws1gt_3,113,149,230).  visible(mws1gt_3,230). happensAt(walking(mws1gt_3),230). orientation(mws1gt_6,146,230). coords(mws1gt_6,48,102,230).  visible(mws1gt_6,230). happensAt(walking(mws1gt_6),230).
orientation(mws1gt_3,162,231). coords(mws1gt_3,113,149,231).  visible(mws1gt_3,231). happensAt(walking(mws1gt_3),231). orientation(mws1gt_6,146,231). coords(mws1gt_6,48,103,231).  visible(mws1gt_6,231). happensAt(walking(mws1gt_6),231).
orientation(mws1gt_3,162,232). coords(mws1gt_3,113,149,232).  visible(mws1gt_3,232). happensAt(walking(mws1gt_3),232). orientation(mws1gt_6,146,232). coords(mws1gt_6,48,103,232).  visible(mws1gt_6,232). happensAt(walking(mws1gt_6),232).
orientation(mws1gt_3,162,233). coords(mws1gt_3,112,148,233).  visible(mws1gt_3,233). happensAt(walking(mws1gt_3),233). orientation(mws1gt_6,146,233). coords(mws1gt_6,48,103,233).  visible(mws1gt_6,233). happensAt(walking(mws1gt_6),233).
orientation(mws1gt_3,162,234). coords(mws1gt_3,112,148,234).  visible(mws1gt_3,234). happensAt(walking(mws1gt_3),234). orientation(mws1gt_6,146,234). coords(mws1gt_6,48,105,234).  visible(mws1gt_6,234). happensAt(walking(mws1gt_6),234).
orientation(mws1gt_3,162,235). coords(mws1gt_3,112,148,235).  visible(mws1gt_3,235). happensAt(walking(mws1gt_3),235). orientation(mws1gt_6,146,235). coords(mws1gt_6,48,106,235).  visible(mws1gt_6,235). happensAt(walking(mws1gt_6),235).
orientation(mws1gt_3,157,236). coords(mws1gt_3,111,148,236).  visible(mws1gt_3,236). happensAt(walking(mws1gt_3),236). orientation(mws1gt_6,146,236). coords(mws1gt_6,48,106,236).  visible(mws1gt_6,236). happensAt(walking(mws1gt_6),236).
orientation(mws1gt_3,157,237). coords(mws1gt_3,110,147,237).  visible(mws1gt_3,237). happensAt(walking(mws1gt_3),237). orientation(mws1gt_6,146,237). coords(mws1gt_6,44,107,237).  visible(mws1gt_6,237). happensAt(walking(mws1gt_6),237).
orientation(mws1gt_3,157,238). coords(mws1gt_3,109,147,238).  visible(mws1gt_3,238). happensAt(walking(mws1gt_3),238). orientation(mws1gt_6,146,238). coords(mws1gt_6,44,108,238).  visible(mws1gt_6,238). happensAt(walking(mws1gt_6),238).
orientation(mws1gt_3,157,239). coords(mws1gt_3,107,147,239).  visible(mws1gt_3,239). happensAt(walking(mws1gt_3),239). orientation(mws1gt_6,146,239). coords(mws1gt_6,44,109,239).  visible(mws1gt_6,239). happensAt(walking(mws1gt_6),239).
orientation(mws1gt_3,157,240). coords(mws1gt_3,106,146,240).  visible(mws1gt_3,240). happensAt(walking(mws1gt_3),240). orientation(mws1gt_6,146,240). coords(mws1gt_6,44,110,240).  visible(mws1gt_6,240). happensAt(walking(mws1gt_6),240).
orientation(mws1gt_3,157,241). coords(mws1gt_3,104,146,241).  visible(mws1gt_3,241). happensAt(walking(mws1gt_3),241). orientation(mws1gt_6,146,241). coords(mws1gt_6,44,111,241).  visible(mws1gt_6,241). happensAt(walking(mws1gt_6),241).
orientation(mws1gt_3,157,242). coords(mws1gt_3,103,144,242).  visible(mws1gt_3,242). happensAt(walking(mws1gt_3),242). orientation(mws1gt_6,146,242). coords(mws1gt_6,43,112,242).  visible(mws1gt_6,242). happensAt(walking(mws1gt_6),242).
orientation(mws1gt_3,157,243). coords(mws1gt_3,103,143,243).  visible(mws1gt_3,243). happensAt(walking(mws1gt_3),243). orientation(mws1gt_6,146,243). coords(mws1gt_6,43,112,243).  visible(mws1gt_6,243). happensAt(walking(mws1gt_6),243).
orientation(mws1gt_3,157,244). coords(mws1gt_3,103,142,244).  visible(mws1gt_3,244). happensAt(walking(mws1gt_3),244). orientation(mws1gt_6,146,244). coords(mws1gt_6,43,112,244).  visible(mws1gt_6,244). happensAt(walking(mws1gt_6),244).
orientation(mws1gt_3,157,245). coords(mws1gt_3,102,141,245).  visible(mws1gt_3,245). happensAt(walking(mws1gt_3),245). orientation(mws1gt_6,146,245). coords(mws1gt_6,43,112,245).  visible(mws1gt_6,245). happensAt(walking(mws1gt_6),245).
orientation(mws1gt_3,157,246). coords(mws1gt_3,102,141,246).  visible(mws1gt_3,246). happensAt(walking(mws1gt_3),246). orientation(mws1gt_6,146,246). coords(mws1gt_6,43,112,246).  visible(mws1gt_6,246). happensAt(walking(mws1gt_6),246).
orientation(mws1gt_3,157,247). coords(mws1gt_3,102,141,247).  visible(mws1gt_3,247). happensAt(walking(mws1gt_3),247). orientation(mws1gt_6,146,247). coords(mws1gt_6,43,112,247).  visible(mws1gt_6,247). happensAt(walking(mws1gt_6),247).
orientation(mws1gt_3,157,248). coords(mws1gt_3,102,141,248).  visible(mws1gt_3,248). happensAt(walking(mws1gt_3),248). orientation(mws1gt_6,146,248). coords(mws1gt_6,43,112,248).  visible(mws1gt_6,248). happensAt(walking(mws1gt_6),248).
orientation(mws1gt_3,157,249). coords(mws1gt_3,102,142,249).  visible(mws1gt_3,249). happensAt(walking(mws1gt_3),249). orientation(mws1gt_6,146,249). coords(mws1gt_6,43,114,249).  visible(mws1gt_6,249). happensAt(walking(mws1gt_6),249).
orientation(mws1gt_3,157,250). coords(mws1gt_3,101,141,250).  visible(mws1gt_3,250). happensAt(walking(mws1gt_3),250). orientation(mws1gt_6,146,250). coords(mws1gt_6,43,115,250).  visible(mws1gt_6,250). happensAt(walking(mws1gt_6),250).
orientation(mws1gt_3,157,251). coords(mws1gt_3,100,141,251).  visible(mws1gt_3,251). happensAt(walking(mws1gt_3),251). orientation(mws1gt_6,146,251). coords(mws1gt_6,42,116,251).  visible(mws1gt_6,251). happensAt(walking(mws1gt_6),251).
orientation(mws1gt_3,157,252). coords(mws1gt_3,100,140,252).  visible(mws1gt_3,252). happensAt(walking(mws1gt_3),252). orientation(mws1gt_6,146,252). coords(mws1gt_6,42,117,252).  visible(mws1gt_6,252). happensAt(walking(mws1gt_6),252).
#show holdsAt/2.