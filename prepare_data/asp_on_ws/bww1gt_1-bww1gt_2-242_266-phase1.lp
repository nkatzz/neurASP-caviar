
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
orientation(bww1gt_1,176,255). coords(bww1gt_1,85,253,255).  visible(bww1gt_1,255). happensAt(walking(bww1gt_1),255). orientation(bww1gt_2,167,255). coords(bww1gt_2,42,165,255).  visible(bww1gt_2,255). happensAt(walking(bww1gt_2),255).
orientation(bww1gt_1,1,256). coords(bww1gt_1,85,253,256).  visible(bww1gt_1,256). happensAt(walking(bww1gt_1),256). orientation(bww1gt_2,167,256). coords(bww1gt_2,43,165,256).  visible(bww1gt_2,256). happensAt(walking(bww1gt_2),256).
orientation(bww1gt_1,178,257). coords(bww1gt_1,86,252,257).  visible(bww1gt_1,257). happensAt(walking(bww1gt_1),257). orientation(bww1gt_2,167,257). coords(bww1gt_2,44,165,257).  visible(bww1gt_2,257). happensAt(walking(bww1gt_2),257).
orientation(bww1gt_1,171,258). coords(bww1gt_1,86,252,258).  visible(bww1gt_1,258). happensAt(walking(bww1gt_1),258). orientation(bww1gt_2,167,258). coords(bww1gt_2,45,165,258).  visible(bww1gt_2,258). happensAt(walking(bww1gt_2),258).
orientation(bww1gt_1,176,259). coords(bww1gt_1,87,252,259).  visible(bww1gt_1,259). happensAt(walking(bww1gt_1),259). orientation(bww1gt_2,167,259). coords(bww1gt_2,46,165,259).  visible(bww1gt_2,259). happensAt(walking(bww1gt_2),259).
orientation(bww1gt_1,172,260). coords(bww1gt_1,87,251,260).  visible(bww1gt_1,260). happensAt(walking(bww1gt_1),260). orientation(bww1gt_2,167,260). coords(bww1gt_2,46,165,260).  visible(bww1gt_2,260). happensAt(walking(bww1gt_2),260).
orientation(bww1gt_1,173,261). coords(bww1gt_1,87,250,261).  visible(bww1gt_1,261). happensAt(walking(bww1gt_1),261). orientation(bww1gt_2,167,261). coords(bww1gt_2,46,165,261).  visible(bww1gt_2,261). happensAt(walking(bww1gt_2),261).
orientation(bww1gt_1,164,262). coords(bww1gt_1,88,250,262).  visible(bww1gt_1,262). happensAt(walking(bww1gt_1),262). orientation(bww1gt_2,167,262). coords(bww1gt_2,46,165,262).  visible(bww1gt_2,262). happensAt(walking(bww1gt_2),262).
orientation(bww1gt_1,170,263). coords(bww1gt_1,88,249,263).  visible(bww1gt_1,263). happensAt(walking(bww1gt_1),263). orientation(bww1gt_2,167,263). coords(bww1gt_2,47,165,263).  visible(bww1gt_2,263). happensAt(walking(bww1gt_2),263).
orientation(bww1gt_1,169,264). coords(bww1gt_1,88,248,264).  visible(bww1gt_1,264). happensAt(walking(bww1gt_1),264). orientation(bww1gt_2,167,264). coords(bww1gt_2,47,165,264).  visible(bww1gt_2,264). happensAt(walking(bww1gt_2),264).
orientation(bww1gt_1,165,265). coords(bww1gt_1,89,248,265).  visible(bww1gt_1,265). happensAt(walking(bww1gt_1),265). orientation(bww1gt_2,167,265). coords(bww1gt_2,47,165,265).  visible(bww1gt_2,265). happensAt(walking(bww1gt_2),265).
orientation(bww1gt_1,168,266). coords(bww1gt_1,89,248,266).  visible(bww1gt_1,266). happensAt(walking(bww1gt_1),266). orientation(bww1gt_2,167,266). coords(bww1gt_2,47,165,266).  visible(bww1gt_2,266). happensAt(walking(bww1gt_2),266).
#show holdsAt/2.