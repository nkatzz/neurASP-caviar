
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

	orientation(bww1gt_1,170,230). coords(bww1gt_1,88,259,230).  visible(bww1gt_1,230). happensAt(walking(bww1gt_1),230). orientation(bww1gt_2,167,230). coords(bww1gt_2,38,166,230).  visible(bww1gt_2,230). happensAt(walking(bww1gt_2),230).
orientation(bww1gt_1,172,231). coords(bww1gt_1,87,258,231).  visible(bww1gt_1,231). happensAt(walking(bww1gt_1),231). orientation(bww1gt_2,167,231). coords(bww1gt_2,38,166,231).  visible(bww1gt_2,231). happensAt(walking(bww1gt_2),231).
orientation(bww1gt_1,172,232). coords(bww1gt_1,87,257,232).  visible(bww1gt_1,232). happensAt(walking(bww1gt_1),232). orientation(bww1gt_2,167,232). coords(bww1gt_2,39,166,232).  visible(bww1gt_2,232). happensAt(walking(bww1gt_2),232).
orientation(bww1gt_1,176,233). coords(bww1gt_1,86,257,233).  visible(bww1gt_1,233). happensAt(walking(bww1gt_1),233). orientation(bww1gt_2,167,233). coords(bww1gt_2,39,166,233).  visible(bww1gt_2,233). happensAt(walking(bww1gt_2),233).
orientation(bww1gt_1,174,234). coords(bww1gt_1,86,257,234).  visible(bww1gt_1,234). happensAt(walking(bww1gt_1),234). orientation(bww1gt_2,167,234). coords(bww1gt_2,39,166,234).  visible(bww1gt_2,234). happensAt(walking(bww1gt_2),234).
orientation(bww1gt_1,170,235). coords(bww1gt_1,86,257,235).  visible(bww1gt_1,235). happensAt(walking(bww1gt_1),235). orientation(bww1gt_2,167,235). coords(bww1gt_2,39,165,235).  visible(bww1gt_2,235). happensAt(walking(bww1gt_2),235).
orientation(bww1gt_1,169,236). coords(bww1gt_1,86,257,236).  visible(bww1gt_1,236). happensAt(walking(bww1gt_1),236). orientation(bww1gt_2,167,236). coords(bww1gt_2,39,165,236).  visible(bww1gt_2,236). happensAt(walking(bww1gt_2),236).
orientation(bww1gt_1,173,237). coords(bww1gt_1,86,257,237).  visible(bww1gt_1,237). happensAt(walking(bww1gt_1),237). orientation(bww1gt_2,167,237). coords(bww1gt_2,39,165,237).  visible(bww1gt_2,237). happensAt(walking(bww1gt_2),237).
orientation(bww1gt_1,172,238). coords(bww1gt_1,86,257,238).  visible(bww1gt_1,238). happensAt(walking(bww1gt_1),238). orientation(bww1gt_2,167,238). coords(bww1gt_2,39,165,238).  visible(bww1gt_2,238). happensAt(walking(bww1gt_2),238).
orientation(bww1gt_1,174,239). coords(bww1gt_1,86,257,239).  visible(bww1gt_1,239). happensAt(walking(bww1gt_1),239). orientation(bww1gt_2,167,239). coords(bww1gt_2,39,165,239).  visible(bww1gt_2,239). happensAt(walking(bww1gt_2),239).
orientation(bww1gt_1,177,240). coords(bww1gt_1,85,257,240).  visible(bww1gt_1,240). happensAt(walking(bww1gt_1),240). orientation(bww1gt_2,167,240). coords(bww1gt_2,39,165,240).  visible(bww1gt_2,240). happensAt(walking(bww1gt_2),240).
orientation(bww1gt_1,5,241). coords(bww1gt_1,85,256,241).  visible(bww1gt_1,241). happensAt(walking(bww1gt_1),241). orientation(bww1gt_2,167,241). coords(bww1gt_2,40,165,241).  visible(bww1gt_2,241). happensAt(walking(bww1gt_2),241).
orientation(bww1gt_1,7,242). coords(bww1gt_1,85,256,242).  visible(bww1gt_1,242). happensAt(walking(bww1gt_1),242). orientation(bww1gt_2,167,242). coords(bww1gt_2,40,165,242).  visible(bww1gt_2,242). happensAt(walking(bww1gt_2),242).
orientation(bww1gt_1,8,243). coords(bww1gt_1,85,256,243).  visible(bww1gt_1,243). happensAt(walking(bww1gt_1),243). orientation(bww1gt_2,167,243). coords(bww1gt_2,41,165,243).  visible(bww1gt_2,243). happensAt(walking(bww1gt_2),243).
orientation(bww1gt_1,11,244). coords(bww1gt_1,85,256,244).  visible(bww1gt_1,244). happensAt(walking(bww1gt_1),244). orientation(bww1gt_2,167,244). coords(bww1gt_2,41,165,244).  visible(bww1gt_2,244). happensAt(walking(bww1gt_2),244).
orientation(bww1gt_1,11,245). coords(bww1gt_1,85,255,245).  visible(bww1gt_1,245). happensAt(walking(bww1gt_1),245). orientation(bww1gt_2,167,245). coords(bww1gt_2,41,165,245).  visible(bww1gt_2,245). happensAt(walking(bww1gt_2),245).
orientation(bww1gt_1,9,246). coords(bww1gt_1,85,255,246).  visible(bww1gt_1,246). happensAt(walking(bww1gt_1),246). orientation(bww1gt_2,167,246). coords(bww1gt_2,42,165,246).  visible(bww1gt_2,246). happensAt(walking(bww1gt_2),246).
orientation(bww1gt_1,12,247). coords(bww1gt_1,85,255,247).  visible(bww1gt_1,247). happensAt(walking(bww1gt_1),247). orientation(bww1gt_2,167,247). coords(bww1gt_2,42,165,247).  visible(bww1gt_2,247). happensAt(walking(bww1gt_2),247).
orientation(bww1gt_1,13,248). coords(bww1gt_1,85,255,248).  visible(bww1gt_1,248). happensAt(walking(bww1gt_1),248). orientation(bww1gt_2,167,248). coords(bww1gt_2,42,165,248).  visible(bww1gt_2,248). happensAt(walking(bww1gt_2),248).
orientation(bww1gt_1,13,249). coords(bww1gt_1,85,255,249).  visible(bww1gt_1,249). happensAt(walking(bww1gt_1),249). orientation(bww1gt_2,167,249). coords(bww1gt_2,42,165,249).  visible(bww1gt_2,249). happensAt(walking(bww1gt_2),249).
orientation(bww1gt_1,10,250). coords(bww1gt_1,85,254,250).  visible(bww1gt_1,250). happensAt(walking(bww1gt_1),250). orientation(bww1gt_2,167,250). coords(bww1gt_2,42,165,250).  visible(bww1gt_2,250). happensAt(walking(bww1gt_2),250).
orientation(bww1gt_1,6,251). coords(bww1gt_1,85,254,251).  visible(bww1gt_1,251). happensAt(walking(bww1gt_1),251). orientation(bww1gt_2,167,251). coords(bww1gt_2,42,165,251).  visible(bww1gt_2,251). happensAt(walking(bww1gt_2),251).
orientation(bww1gt_1,179,252). coords(bww1gt_1,85,254,252).  visible(bww1gt_1,252). happensAt(walking(bww1gt_1),252). orientation(bww1gt_2,167,252). coords(bww1gt_2,42,165,252).  visible(bww1gt_2,252). happensAt(walking(bww1gt_2),252).
orientation(bww1gt_1,179,253). coords(bww1gt_1,85,254,253).  visible(bww1gt_1,253). happensAt(walking(bww1gt_1),253). orientation(bww1gt_2,167,253). coords(bww1gt_2,42,165,253).  visible(bww1gt_2,253). happensAt(walking(bww1gt_2),253).
orientation(bww1gt_1,176,254). coords(bww1gt_1,85,254,254).  visible(bww1gt_1,254). happensAt(walking(bww1gt_1),254). orientation(bww1gt_2,167,254). coords(bww1gt_2,42,165,254).  visible(bww1gt_2,254). happensAt(walking(bww1gt_2),254).
#show holdsAt/2.