
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

	orientation(fcgt_1,154,244). coords(fcgt_1,171,162,244).  visible(fcgt_1,244). happensAt(walking(fcgt_1),244). orientation(fcgt_3,170,244). coords(fcgt_3,101,175,244).  visible(fcgt_3,244). happensAt(walking(fcgt_3),244).
orientation(fcgt_1,154,245). coords(fcgt_1,170,161,245).  visible(fcgt_1,245). happensAt(walking(fcgt_1),245). orientation(fcgt_3,170,245). coords(fcgt_3,101,175,245).  visible(fcgt_3,245). happensAt(walking(fcgt_3),245).
orientation(fcgt_1,154,246). coords(fcgt_1,169,160,246).  visible(fcgt_1,246). happensAt(walking(fcgt_1),246). orientation(fcgt_3,170,246). coords(fcgt_3,101,176,246).  visible(fcgt_3,246). happensAt(walking(fcgt_3),246).
orientation(fcgt_1,154,247). coords(fcgt_1,167,159,247).  visible(fcgt_1,247). happensAt(walking(fcgt_1),247). orientation(fcgt_3,170,247). coords(fcgt_3,101,176,247).  visible(fcgt_3,247). happensAt(walking(fcgt_3),247).
orientation(fcgt_1,154,248). coords(fcgt_1,165,157,248).  visible(fcgt_1,248). happensAt(walking(fcgt_1),248). orientation(fcgt_3,170,248). coords(fcgt_3,101,176,248).  visible(fcgt_3,248). happensAt(walking(fcgt_3),248).
orientation(fcgt_1,157,249). coords(fcgt_1,164,155,249).  visible(fcgt_1,249). happensAt(walking(fcgt_1),249). orientation(fcgt_3,168,249). coords(fcgt_3,101,176,249).  visible(fcgt_3,249). happensAt(walking(fcgt_3),249).
orientation(fcgt_1,157,250). coords(fcgt_1,159,154,250).  visible(fcgt_1,250). happensAt(walking(fcgt_1),250). orientation(fcgt_3,167,250). coords(fcgt_3,101,176,250).  visible(fcgt_3,250). happensAt(walking(fcgt_3),250).
orientation(fcgt_1,157,251). coords(fcgt_1,157,152,251).  visible(fcgt_1,251). happensAt(walking(fcgt_1),251). orientation(fcgt_3,165,251). coords(fcgt_3,101,176,251).  visible(fcgt_3,251). happensAt(walking(fcgt_3),251).
orientation(fcgt_1,157,252). coords(fcgt_1,154,153,252).  visible(fcgt_1,252). happensAt(walking(fcgt_1),252). orientation(fcgt_3,165,252). coords(fcgt_3,101,177,252).  visible(fcgt_3,252). happensAt(walking(fcgt_3),252).
orientation(fcgt_1,157,253). coords(fcgt_1,154,152,253).  visible(fcgt_1,253). happensAt(walking(fcgt_1),253). orientation(fcgt_3,165,253). coords(fcgt_3,101,177,253).  visible(fcgt_3,253). happensAt(walking(fcgt_3),253).
orientation(fcgt_1,157,254). coords(fcgt_1,153,151,254).  visible(fcgt_1,254). happensAt(walking(fcgt_1),254). orientation(fcgt_3,166,254). coords(fcgt_3,100,177,254).  visible(fcgt_3,254). happensAt(walking(fcgt_3),254).
orientation(fcgt_1,157,255). coords(fcgt_1,152,151,255).  visible(fcgt_1,255). happensAt(walking(fcgt_1),255). orientation(fcgt_3,166,255). coords(fcgt_3,100,177,255).  visible(fcgt_3,255). happensAt(walking(fcgt_3),255).
orientation(fcgt_1,157,256). coords(fcgt_1,152,150,256).  visible(fcgt_1,256). happensAt(walking(fcgt_1),256). orientation(fcgt_3,165,256). coords(fcgt_3,100,178,256).  visible(fcgt_3,256). happensAt(walking(fcgt_3),256).
orientation(fcgt_1,151,257). coords(fcgt_1,151,150,257).  visible(fcgt_1,257). happensAt(walking(fcgt_1),257). orientation(fcgt_3,167,257). coords(fcgt_3,100,178,257).  visible(fcgt_3,257). happensAt(walking(fcgt_3),257).
orientation(fcgt_1,151,258). coords(fcgt_1,149,149,258).  visible(fcgt_1,258). happensAt(walking(fcgt_1),258). orientation(fcgt_3,166,258). coords(fcgt_3,100,178,258).  visible(fcgt_3,258). happensAt(walking(fcgt_3),258).
orientation(fcgt_1,151,259). coords(fcgt_1,148,148,259).  visible(fcgt_1,259). happensAt(walking(fcgt_1),259). orientation(fcgt_3,164,259). coords(fcgt_3,100,179,259).  visible(fcgt_3,259). happensAt(walking(fcgt_3),259).
orientation(fcgt_1,151,260). coords(fcgt_1,145,146,260).  visible(fcgt_1,260). happensAt(walking(fcgt_1),260). orientation(fcgt_3,165,260). coords(fcgt_3,100,179,260).  visible(fcgt_3,260). happensAt(walking(fcgt_3),260).
orientation(fcgt_1,151,261). coords(fcgt_1,143,145,261).  visible(fcgt_1,261). happensAt(walking(fcgt_1),261). orientation(fcgt_3,163,261). coords(fcgt_3,100,179,261).  visible(fcgt_3,261). happensAt(walking(fcgt_3),261).
orientation(fcgt_1,151,262). coords(fcgt_1,140,143,262).  visible(fcgt_1,262). happensAt(walking(fcgt_1),262). orientation(fcgt_3,166,262). coords(fcgt_3,100,179,262).  visible(fcgt_3,262). happensAt(walking(fcgt_3),262).
orientation(fcgt_1,151,263). coords(fcgt_1,139,141,263).  visible(fcgt_1,263). happensAt(walking(fcgt_1),263). orientation(fcgt_3,163,263). coords(fcgt_3,100,180,263).  visible(fcgt_3,263). happensAt(walking(fcgt_3),263).
orientation(fcgt_1,151,264). coords(fcgt_1,138,139,264).  visible(fcgt_1,264). happensAt(walking(fcgt_1),264). orientation(fcgt_3,165,264). coords(fcgt_3,100,180,264).  visible(fcgt_3,264). happensAt(walking(fcgt_3),264).
orientation(fcgt_1,151,265). coords(fcgt_1,138,138,265).  visible(fcgt_1,265). happensAt(walking(fcgt_1),265). orientation(fcgt_3,164,265). coords(fcgt_3,99,180,265).  visible(fcgt_3,265). happensAt(walking(fcgt_3),265).
orientation(fcgt_1,151,266). coords(fcgt_1,137,138,266).  visible(fcgt_1,266). happensAt(walking(fcgt_1),266). orientation(fcgt_3,163,266). coords(fcgt_3,99,181,266).  visible(fcgt_3,266). happensAt(walking(fcgt_3),266).
orientation(fcgt_1,151,267). coords(fcgt_1,136,137,267).  visible(fcgt_1,267). happensAt(walking(fcgt_1),267). orientation(fcgt_3,161,267). coords(fcgt_3,99,181,267).  visible(fcgt_3,267). happensAt(walking(fcgt_3),267).
orientation(fcgt_1,151,268). coords(fcgt_1,135,137,268).  visible(fcgt_1,268). happensAt(walking(fcgt_1),268). orientation(fcgt_3,163,268). coords(fcgt_3,99,181,268).  visible(fcgt_3,268). happensAt(walking(fcgt_3),268).
#show holdsAt/2.