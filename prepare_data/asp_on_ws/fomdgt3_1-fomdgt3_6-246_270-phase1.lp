
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

	orientation(fomdgt3_1,150,246). coords(fomdgt3_1,64,93,246).  visible(fomdgt3_1,246). happensAt(walking(fomdgt3_1),246). orientation(fomdgt3_6,130,246). coords(fomdgt3_6,209,133,246).  visible(fomdgt3_6,246). happensAt(walking(fomdgt3_6),246).
orientation(fomdgt3_1,150,247). coords(fomdgt3_1,64,93,247).  visible(fomdgt3_1,247). happensAt(walking(fomdgt3_1),247). orientation(fomdgt3_6,130,247). coords(fomdgt3_6,209,135,247).  visible(fomdgt3_6,247). happensAt(walking(fomdgt3_6),247).
orientation(fomdgt3_1,150,248). coords(fomdgt3_1,64,93,248).  visible(fomdgt3_1,248). happensAt(walking(fomdgt3_1),248). orientation(fomdgt3_6,130,248). coords(fomdgt3_6,209,135,248).  visible(fomdgt3_6,248). happensAt(walking(fomdgt3_6),248).
orientation(fomdgt3_1,150,249). coords(fomdgt3_1,61,94,249).  visible(fomdgt3_1,249). happensAt(walking(fomdgt3_1),249). orientation(fomdgt3_6,130,249). coords(fomdgt3_6,203,137,249).  visible(fomdgt3_6,249). happensAt(walking(fomdgt3_6),249).
orientation(fomdgt3_1,150,250). coords(fomdgt3_1,61,94,250).  visible(fomdgt3_1,250). happensAt(walking(fomdgt3_1),250). orientation(fomdgt3_6,130,250). coords(fomdgt3_6,203,137,250).  visible(fomdgt3_6,250). happensAt(walking(fomdgt3_6),250).
orientation(fomdgt3_1,150,251). coords(fomdgt3_1,61,94,251).  visible(fomdgt3_1,251). happensAt(walking(fomdgt3_1),251). orientation(fomdgt3_6,130,251). coords(fomdgt3_6,203,140,251).  visible(fomdgt3_6,251). happensAt(walking(fomdgt3_6),251).
orientation(fomdgt3_1,150,252). coords(fomdgt3_1,61,94,252).  visible(fomdgt3_1,252). happensAt(walking(fomdgt3_1),252). orientation(fomdgt3_6,130,252). coords(fomdgt3_6,203,142,252).  visible(fomdgt3_6,252). happensAt(walking(fomdgt3_6),252).
orientation(fomdgt3_1,150,253). coords(fomdgt3_1,61,94,253).  visible(fomdgt3_1,253). happensAt(walking(fomdgt3_1),253). orientation(fomdgt3_6,130,253). coords(fomdgt3_6,203,144,253).  visible(fomdgt3_6,253). happensAt(walking(fomdgt3_6),253).
orientation(fomdgt3_1,150,254). coords(fomdgt3_1,61,94,254).  visible(fomdgt3_1,254). happensAt(walking(fomdgt3_1),254). orientation(fomdgt3_6,130,254). coords(fomdgt3_6,203,140,254).  visible(fomdgt3_6,254). happensAt(walking(fomdgt3_6),254).
orientation(fomdgt3_1,150,255). coords(fomdgt3_1,61,94,255).  visible(fomdgt3_1,255). happensAt(walking(fomdgt3_1),255). orientation(fomdgt3_6,130,255). coords(fomdgt3_6,205,146,255).  visible(fomdgt3_6,255). happensAt(walking(fomdgt3_6),255).
orientation(fomdgt3_1,150,256). coords(fomdgt3_1,61,94,256).  visible(fomdgt3_1,256). happensAt(walking(fomdgt3_1),256). orientation(fomdgt3_6,130,256). coords(fomdgt3_6,205,146,256).  visible(fomdgt3_6,256). happensAt(walking(fomdgt3_6),256).
orientation(fomdgt3_1,150,257). coords(fomdgt3_1,61,94,257).  visible(fomdgt3_1,257). happensAt(walking(fomdgt3_1),257). orientation(fomdgt3_6,130,257). coords(fomdgt3_6,206,150,257).  visible(fomdgt3_6,257). happensAt(walking(fomdgt3_6),257).
orientation(fomdgt3_1,150,258). coords(fomdgt3_1,61,94,258).  visible(fomdgt3_1,258). happensAt(walking(fomdgt3_1),258). orientation(fomdgt3_6,130,258). coords(fomdgt3_6,206,150,258).  visible(fomdgt3_6,258). happensAt(walking(fomdgt3_6),258).
orientation(fomdgt3_1,150,259). coords(fomdgt3_1,61,94,259).  visible(fomdgt3_1,259). happensAt(walking(fomdgt3_1),259). orientation(fomdgt3_6,130,259). coords(fomdgt3_6,206,150,259).  visible(fomdgt3_6,259). happensAt(walking(fomdgt3_6),259).
orientation(fomdgt3_1,150,260). coords(fomdgt3_1,61,94,260).  visible(fomdgt3_1,260). happensAt(walking(fomdgt3_1),260). orientation(fomdgt3_6,130,260). coords(fomdgt3_6,206,156,260).  visible(fomdgt3_6,260). happensAt(walking(fomdgt3_6),260).
orientation(fomdgt3_1,150,261). coords(fomdgt3_1,59,96,261).  visible(fomdgt3_1,261). happensAt(walking(fomdgt3_1),261). orientation(fomdgt3_6,130,261). coords(fomdgt3_6,209,156,261).  visible(fomdgt3_6,261). happensAt(walking(fomdgt3_6),261).
orientation(fomdgt3_1,150,262). coords(fomdgt3_1,59,96,262).  visible(fomdgt3_1,262). happensAt(walking(fomdgt3_1),262). orientation(fomdgt3_6,130,262). coords(fomdgt3_6,209,156,262).  visible(fomdgt3_6,262). happensAt(walking(fomdgt3_6),262).
orientation(fomdgt3_1,150,263). coords(fomdgt3_1,59,96,263).  visible(fomdgt3_1,263). happensAt(walking(fomdgt3_1),263). orientation(fomdgt3_6,130,263). coords(fomdgt3_6,214,161,263).  visible(fomdgt3_6,263). happensAt(walking(fomdgt3_6),263).
orientation(fomdgt3_1,150,264). coords(fomdgt3_1,59,96,264).  visible(fomdgt3_1,264). happensAt(walking(fomdgt3_1),264). orientation(fomdgt3_6,130,264). coords(fomdgt3_6,218,162,264).  visible(fomdgt3_6,264). happensAt(walking(fomdgt3_6),264).
orientation(fomdgt3_1,150,265). coords(fomdgt3_1,59,96,265).  visible(fomdgt3_1,265). happensAt(walking(fomdgt3_1),265). orientation(fomdgt3_6,130,265). coords(fomdgt3_6,220,162,265).  visible(fomdgt3_6,265). happensAt(walking(fomdgt3_6),265).
orientation(fomdgt3_1,150,266). coords(fomdgt3_1,59,96,266).  visible(fomdgt3_1,266). happensAt(walking(fomdgt3_1),266). orientation(fomdgt3_6,130,266). coords(fomdgt3_6,220,162,266).  visible(fomdgt3_6,266). happensAt(walking(fomdgt3_6),266).
orientation(fomdgt3_1,150,267). coords(fomdgt3_1,58,96,267).  visible(fomdgt3_1,267). happensAt(walking(fomdgt3_1),267). orientation(fomdgt3_6,130,267). coords(fomdgt3_6,223,164,267).  visible(fomdgt3_6,267). happensAt(walking(fomdgt3_6),267).
orientation(fomdgt3_1,150,268). coords(fomdgt3_1,58,96,268).  visible(fomdgt3_1,268). happensAt(walking(fomdgt3_1),268). orientation(fomdgt3_6,130,268). coords(fomdgt3_6,223,164,268).  visible(fomdgt3_6,268). happensAt(walking(fomdgt3_6),268).
orientation(fomdgt3_1,150,269). coords(fomdgt3_1,57,96,269).  visible(fomdgt3_1,269). happensAt(walking(fomdgt3_1),269). orientation(fomdgt3_6,130,269). coords(fomdgt3_6,223,164,269).  visible(fomdgt3_6,269). happensAt(walking(fomdgt3_6),269).
orientation(fomdgt3_1,150,270). coords(fomdgt3_1,57,96,270).  visible(fomdgt3_1,270). happensAt(walking(fomdgt3_1),270). orientation(fomdgt3_6,130,270). coords(fomdgt3_6,223,164,270).  visible(fomdgt3_6,270). happensAt(walking(fomdgt3_6),270).
#show holdsAt/2.