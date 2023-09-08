
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

	orientation(wk1gt_4,162,231). coords(wk1gt_4,47,181,231).  visible(wk1gt_4,231). happensAt(inactive(wk1gt_4),231). orientation(wk1gt_5,162,231). coords(wk1gt_5,50,166,231).  visible(wk1gt_5,231). happensAt(inactive(wk1gt_5),231).
orientation(wk1gt_4,160,232). coords(wk1gt_4,47,181,232).  visible(wk1gt_4,232). happensAt(inactive(wk1gt_4),232). orientation(wk1gt_5,162,232). coords(wk1gt_5,50,166,232).  visible(wk1gt_5,232). happensAt(inactive(wk1gt_5),232).
orientation(wk1gt_4,163,233). coords(wk1gt_4,47,181,233).  visible(wk1gt_4,233). happensAt(inactive(wk1gt_4),233). orientation(wk1gt_5,162,233). coords(wk1gt_5,50,166,233).  visible(wk1gt_5,233). happensAt(inactive(wk1gt_5),233).
orientation(wk1gt_4,163,234). coords(wk1gt_4,47,181,234).  visible(wk1gt_4,234). happensAt(inactive(wk1gt_4),234). orientation(wk1gt_5,162,234). coords(wk1gt_5,50,166,234).  visible(wk1gt_5,234). happensAt(inactive(wk1gt_5),234).
orientation(wk1gt_4,166,235). coords(wk1gt_4,47,181,235).  visible(wk1gt_4,235). happensAt(inactive(wk1gt_4),235). orientation(wk1gt_5,162,235). coords(wk1gt_5,50,166,235).  visible(wk1gt_5,235). happensAt(inactive(wk1gt_5),235).
orientation(wk1gt_4,165,236). coords(wk1gt_4,47,181,236).  visible(wk1gt_4,236). happensAt(inactive(wk1gt_4),236). orientation(wk1gt_5,162,236). coords(wk1gt_5,50,166,236).  visible(wk1gt_5,236). happensAt(inactive(wk1gt_5),236).
orientation(wk1gt_4,162,237). coords(wk1gt_4,47,181,237).  visible(wk1gt_4,237). happensAt(inactive(wk1gt_4),237). orientation(wk1gt_5,162,237). coords(wk1gt_5,50,166,237).  visible(wk1gt_5,237). happensAt(inactive(wk1gt_5),237).
orientation(wk1gt_4,165,238). coords(wk1gt_4,47,181,238).  visible(wk1gt_4,238). happensAt(inactive(wk1gt_4),238). orientation(wk1gt_5,162,238). coords(wk1gt_5,50,166,238).  visible(wk1gt_5,238). happensAt(inactive(wk1gt_5),238).
orientation(wk1gt_4,163,239). coords(wk1gt_4,47,181,239).  visible(wk1gt_4,239). happensAt(inactive(wk1gt_4),239). orientation(wk1gt_5,162,239). coords(wk1gt_5,50,166,239).  visible(wk1gt_5,239). happensAt(inactive(wk1gt_5),239).
orientation(wk1gt_4,163,240). coords(wk1gt_4,47,181,240).  visible(wk1gt_4,240). happensAt(inactive(wk1gt_4),240). orientation(wk1gt_5,162,240). coords(wk1gt_5,50,166,240).  visible(wk1gt_5,240). happensAt(inactive(wk1gt_5),240).
orientation(wk1gt_4,161,241). coords(wk1gt_4,47,181,241).  visible(wk1gt_4,241). happensAt(inactive(wk1gt_4),241). orientation(wk1gt_5,162,241). coords(wk1gt_5,50,166,241).  visible(wk1gt_5,241). happensAt(inactive(wk1gt_5),241).
orientation(wk1gt_4,162,242). coords(wk1gt_4,47,181,242).  visible(wk1gt_4,242). happensAt(inactive(wk1gt_4),242). orientation(wk1gt_5,162,242). coords(wk1gt_5,50,166,242).  visible(wk1gt_5,242). happensAt(inactive(wk1gt_5),242).
orientation(wk1gt_4,160,243). coords(wk1gt_4,47,181,243).  visible(wk1gt_4,243). happensAt(inactive(wk1gt_4),243). orientation(wk1gt_5,162,243). coords(wk1gt_5,50,166,243).  visible(wk1gt_5,243). happensAt(inactive(wk1gt_5),243).
orientation(wk1gt_4,162,244). coords(wk1gt_4,47,181,244).  visible(wk1gt_4,244). happensAt(inactive(wk1gt_4),244). orientation(wk1gt_5,162,244). coords(wk1gt_5,50,166,244).  visible(wk1gt_5,244). happensAt(inactive(wk1gt_5),244).
orientation(wk1gt_4,161,245). coords(wk1gt_4,47,181,245).  visible(wk1gt_4,245). happensAt(inactive(wk1gt_4),245). orientation(wk1gt_5,162,245). coords(wk1gt_5,50,166,245).  visible(wk1gt_5,245). happensAt(inactive(wk1gt_5),245).
orientation(wk1gt_4,155,246). coords(wk1gt_4,47,181,246).  visible(wk1gt_4,246). happensAt(inactive(wk1gt_4),246). orientation(wk1gt_5,162,246). coords(wk1gt_5,50,166,246).  visible(wk1gt_5,246). happensAt(inactive(wk1gt_5),246).
orientation(wk1gt_4,154,247). coords(wk1gt_4,47,181,247).  visible(wk1gt_4,247). happensAt(inactive(wk1gt_4),247). orientation(wk1gt_5,162,247). coords(wk1gt_5,50,166,247).  visible(wk1gt_5,247). happensAt(inactive(wk1gt_5),247).
orientation(wk1gt_4,163,248). coords(wk1gt_4,47,181,248).  visible(wk1gt_4,248). happensAt(inactive(wk1gt_4),248). orientation(wk1gt_5,162,248). coords(wk1gt_5,50,166,248).  visible(wk1gt_5,248). happensAt(inactive(wk1gt_5),248).
orientation(wk1gt_4,161,249). coords(wk1gt_4,47,181,249).  visible(wk1gt_4,249). happensAt(inactive(wk1gt_4),249). orientation(wk1gt_5,162,249). coords(wk1gt_5,50,166,249).  visible(wk1gt_5,249). happensAt(inactive(wk1gt_5),249).
orientation(wk1gt_4,160,250). coords(wk1gt_4,47,181,250).  visible(wk1gt_4,250). happensAt(inactive(wk1gt_4),250). orientation(wk1gt_5,162,250). coords(wk1gt_5,50,166,250).  visible(wk1gt_5,250). happensAt(inactive(wk1gt_5),250).
orientation(wk1gt_4,162,251). coords(wk1gt_4,47,181,251).  visible(wk1gt_4,251). happensAt(inactive(wk1gt_4),251). orientation(wk1gt_5,162,251). coords(wk1gt_5,50,166,251).  visible(wk1gt_5,251). happensAt(inactive(wk1gt_5),251).
orientation(wk1gt_4,161,252). coords(wk1gt_4,47,181,252).  visible(wk1gt_4,252). happensAt(inactive(wk1gt_4),252). orientation(wk1gt_5,162,252). coords(wk1gt_5,50,166,252).  visible(wk1gt_5,252). happensAt(inactive(wk1gt_5),252).
orientation(wk1gt_4,163,253). coords(wk1gt_4,47,181,253).  visible(wk1gt_4,253). happensAt(active(wk1gt_4),253). orientation(wk1gt_5,162,253). coords(wk1gt_5,50,166,253).  visible(wk1gt_5,253). happensAt(inactive(wk1gt_5),253).
orientation(wk1gt_4,161,254). coords(wk1gt_4,47,181,254).  visible(wk1gt_4,254). happensAt(active(wk1gt_4),254). orientation(wk1gt_5,162,254). coords(wk1gt_5,50,166,254).  visible(wk1gt_5,254). happensAt(inactive(wk1gt_5),254).
orientation(wk1gt_4,163,255). coords(wk1gt_4,47,181,255).  visible(wk1gt_4,255). happensAt(active(wk1gt_4),255). orientation(wk1gt_5,162,255). coords(wk1gt_5,50,166,255).  visible(wk1gt_5,255). happensAt(inactive(wk1gt_5),255).
#show holdsAt/2.