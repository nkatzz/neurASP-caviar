
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

	orientation(wk1gt_1,90,236). coords(wk1gt_1,244,285,236).  happensAt(appear(wk1gt_1),236). happensAt(walking(wk1gt_1),236). orientation(wk1gt_4,165,236). coords(wk1gt_4,47,181,236).  visible(wk1gt_4,236). happensAt(inactive(wk1gt_4),236).
orientation(wk1gt_1,90,237). coords(wk1gt_1,244,284,237).  visible(wk1gt_1,237). happensAt(walking(wk1gt_1),237). orientation(wk1gt_4,162,237). coords(wk1gt_4,47,181,237).  visible(wk1gt_4,237). happensAt(inactive(wk1gt_4),237).
orientation(wk1gt_1,90,238). coords(wk1gt_1,244,284,238).  visible(wk1gt_1,238). happensAt(walking(wk1gt_1),238). orientation(wk1gt_4,165,238). coords(wk1gt_4,47,181,238).  visible(wk1gt_4,238). happensAt(inactive(wk1gt_4),238).
orientation(wk1gt_1,90,239). coords(wk1gt_1,244,283,239).  visible(wk1gt_1,239). happensAt(walking(wk1gt_1),239). orientation(wk1gt_4,163,239). coords(wk1gt_4,47,181,239).  visible(wk1gt_4,239). happensAt(inactive(wk1gt_4),239).
orientation(wk1gt_1,90,240). coords(wk1gt_1,244,283,240).  visible(wk1gt_1,240). happensAt(walking(wk1gt_1),240). orientation(wk1gt_4,163,240). coords(wk1gt_4,47,181,240).  visible(wk1gt_4,240). happensAt(inactive(wk1gt_4),240).
orientation(wk1gt_1,90,241). coords(wk1gt_1,244,283,241).  visible(wk1gt_1,241). happensAt(walking(wk1gt_1),241). orientation(wk1gt_4,161,241). coords(wk1gt_4,47,181,241).  visible(wk1gt_4,241). happensAt(inactive(wk1gt_4),241).
orientation(wk1gt_1,90,242). coords(wk1gt_1,244,283,242).  visible(wk1gt_1,242). happensAt(walking(wk1gt_1),242). orientation(wk1gt_4,162,242). coords(wk1gt_4,47,181,242).  visible(wk1gt_4,242). happensAt(inactive(wk1gt_4),242).
orientation(wk1gt_1,90,243). coords(wk1gt_1,244,283,243).  visible(wk1gt_1,243). happensAt(walking(wk1gt_1),243). orientation(wk1gt_4,160,243). coords(wk1gt_4,47,181,243).  visible(wk1gt_4,243). happensAt(inactive(wk1gt_4),243).
orientation(wk1gt_1,90,244). coords(wk1gt_1,242,283,244).  visible(wk1gt_1,244). happensAt(walking(wk1gt_1),244). orientation(wk1gt_4,162,244). coords(wk1gt_4,47,181,244).  visible(wk1gt_4,244). happensAt(inactive(wk1gt_4),244).
orientation(wk1gt_1,90,245). coords(wk1gt_1,242,283,245).  visible(wk1gt_1,245). happensAt(walking(wk1gt_1),245). orientation(wk1gt_4,161,245). coords(wk1gt_4,47,181,245).  visible(wk1gt_4,245). happensAt(inactive(wk1gt_4),245).
orientation(wk1gt_1,90,246). coords(wk1gt_1,242,283,246).  visible(wk1gt_1,246). happensAt(walking(wk1gt_1),246). orientation(wk1gt_4,155,246). coords(wk1gt_4,47,181,246).  visible(wk1gt_4,246). happensAt(inactive(wk1gt_4),246).
orientation(wk1gt_1,35,247). coords(wk1gt_1,236,281,247).  visible(wk1gt_1,247). happensAt(walking(wk1gt_1),247). orientation(wk1gt_4,154,247). coords(wk1gt_4,47,181,247).  visible(wk1gt_4,247). happensAt(inactive(wk1gt_4),247).
orientation(wk1gt_1,35,248). coords(wk1gt_1,235,278,248).  visible(wk1gt_1,248). happensAt(walking(wk1gt_1),248). orientation(wk1gt_4,163,248). coords(wk1gt_4,47,181,248).  visible(wk1gt_4,248). happensAt(inactive(wk1gt_4),248).
orientation(wk1gt_1,35,249). coords(wk1gt_1,231,275,249).  visible(wk1gt_1,249). happensAt(walking(wk1gt_1),249). orientation(wk1gt_4,161,249). coords(wk1gt_4,47,181,249).  visible(wk1gt_4,249). happensAt(inactive(wk1gt_4),249).
orientation(wk1gt_1,35,250). coords(wk1gt_1,228,273,250).  visible(wk1gt_1,250). happensAt(walking(wk1gt_1),250). orientation(wk1gt_4,160,250). coords(wk1gt_4,47,181,250).  visible(wk1gt_4,250). happensAt(inactive(wk1gt_4),250).
orientation(wk1gt_1,35,251). coords(wk1gt_1,228,272,251).  visible(wk1gt_1,251). happensAt(walking(wk1gt_1),251). orientation(wk1gt_4,162,251). coords(wk1gt_4,47,181,251).  visible(wk1gt_4,251). happensAt(inactive(wk1gt_4),251).
orientation(wk1gt_1,26,252). coords(wk1gt_1,228,272,252).  visible(wk1gt_1,252). happensAt(walking(wk1gt_1),252). orientation(wk1gt_4,161,252). coords(wk1gt_4,47,181,252).  visible(wk1gt_4,252). happensAt(inactive(wk1gt_4),252).
orientation(wk1gt_1,21,253). coords(wk1gt_1,228,272,253).  visible(wk1gt_1,253). happensAt(walking(wk1gt_1),253). orientation(wk1gt_4,163,253). coords(wk1gt_4,47,181,253).  visible(wk1gt_4,253). happensAt(active(wk1gt_4),253).
orientation(wk1gt_1,21,254). coords(wk1gt_1,228,272,254).  visible(wk1gt_1,254). happensAt(walking(wk1gt_1),254). orientation(wk1gt_4,161,254). coords(wk1gt_4,47,181,254).  visible(wk1gt_4,254). happensAt(active(wk1gt_4),254).
orientation(wk1gt_1,11,255). coords(wk1gt_1,227,272,255).  visible(wk1gt_1,255). happensAt(walking(wk1gt_1),255). orientation(wk1gt_4,163,255). coords(wk1gt_4,47,181,255).  visible(wk1gt_4,255). happensAt(active(wk1gt_4),255).
orientation(wk1gt_1,11,256). coords(wk1gt_1,226,270,256).  visible(wk1gt_1,256). happensAt(walking(wk1gt_1),256). orientation(wk1gt_4,161,256). coords(wk1gt_4,47,181,256).  visible(wk1gt_4,256). happensAt(active(wk1gt_4),256).
orientation(wk1gt_1,11,257). coords(wk1gt_1,225,269,257).  visible(wk1gt_1,257). happensAt(walking(wk1gt_1),257). orientation(wk1gt_4,160,257). coords(wk1gt_4,47,181,257).  visible(wk1gt_4,257). happensAt(active(wk1gt_4),257).
orientation(wk1gt_1,11,258). coords(wk1gt_1,225,267,258).  visible(wk1gt_1,258). happensAt(walking(wk1gt_1),258). orientation(wk1gt_4,157,258). coords(wk1gt_4,47,181,258).  visible(wk1gt_4,258). happensAt(active(wk1gt_4),258).
orientation(wk1gt_1,11,259). coords(wk1gt_1,224,267,259).  visible(wk1gt_1,259). happensAt(walking(wk1gt_1),259). orientation(wk1gt_4,160,259). coords(wk1gt_4,47,180,259).  visible(wk1gt_4,259). happensAt(active(wk1gt_4),259).
orientation(wk1gt_1,11,260). coords(wk1gt_1,224,264,260).  visible(wk1gt_1,260). happensAt(walking(wk1gt_1),260). orientation(wk1gt_4,160,260). coords(wk1gt_4,48,180,260).  visible(wk1gt_4,260). happensAt(active(wk1gt_4),260).
#show holdsAt/2.