
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

	orientation(fcgt_2,157,246). coords(fcgt_2,58,161,246).  visible(fcgt_2,246). happensAt(walking(fcgt_2),246). orientation(fcgt_4,166,246). coords(fcgt_4,80,205,246).  visible(fcgt_4,246). happensAt(inactive(fcgt_4),246).
orientation(fcgt_2,157,247). coords(fcgt_2,58,161,247).  visible(fcgt_2,247). happensAt(walking(fcgt_2),247). orientation(fcgt_4,166,247). coords(fcgt_4,80,205,247).  visible(fcgt_4,247). happensAt(inactive(fcgt_4),247).
orientation(fcgt_2,157,248). coords(fcgt_2,59,161,248).  visible(fcgt_2,248). happensAt(walking(fcgt_2),248). orientation(fcgt_4,166,248). coords(fcgt_4,80,205,248).  visible(fcgt_4,248). happensAt(inactive(fcgt_4),248).
orientation(fcgt_2,157,249). coords(fcgt_2,61,160,249).  visible(fcgt_2,249). happensAt(walking(fcgt_2),249). orientation(fcgt_4,166,249). coords(fcgt_4,80,205,249).  visible(fcgt_4,249). happensAt(inactive(fcgt_4),249).
orientation(fcgt_2,157,250). coords(fcgt_2,61,159,250).  visible(fcgt_2,250). happensAt(walking(fcgt_2),250). orientation(fcgt_4,166,250). coords(fcgt_4,80,205,250).  visible(fcgt_4,250). happensAt(inactive(fcgt_4),250).
orientation(fcgt_2,157,251). coords(fcgt_2,62,159,251).  visible(fcgt_2,251). happensAt(walking(fcgt_2),251). orientation(fcgt_4,166,251). coords(fcgt_4,80,205,251).  visible(fcgt_4,251). happensAt(inactive(fcgt_4),251).
orientation(fcgt_2,157,252). coords(fcgt_2,63,158,252).  visible(fcgt_2,252). happensAt(walking(fcgt_2),252). orientation(fcgt_4,166,252). coords(fcgt_4,80,205,252).  visible(fcgt_4,252). happensAt(inactive(fcgt_4),252).
orientation(fcgt_2,157,253). coords(fcgt_2,63,158,253).  visible(fcgt_2,253). happensAt(walking(fcgt_2),253). orientation(fcgt_4,166,253). coords(fcgt_4,80,205,253).  visible(fcgt_4,253). happensAt(inactive(fcgt_4),253).
orientation(fcgt_2,157,254). coords(fcgt_2,63,157,254).  visible(fcgt_2,254). happensAt(walking(fcgt_2),254). orientation(fcgt_4,166,254). coords(fcgt_4,80,205,254).  visible(fcgt_4,254). happensAt(inactive(fcgt_4),254).
orientation(fcgt_2,157,255). coords(fcgt_2,64,156,255).  visible(fcgt_2,255). happensAt(walking(fcgt_2),255). orientation(fcgt_4,166,255). coords(fcgt_4,80,205,255).  visible(fcgt_4,255). happensAt(inactive(fcgt_4),255).
orientation(fcgt_2,157,256). coords(fcgt_2,64,155,256).  visible(fcgt_2,256). happensAt(walking(fcgt_2),256). orientation(fcgt_4,166,256). coords(fcgt_4,80,205,256).  visible(fcgt_4,256). happensAt(inactive(fcgt_4),256).
orientation(fcgt_2,157,257). coords(fcgt_2,64,154,257).  visible(fcgt_2,257). happensAt(walking(fcgt_2),257). orientation(fcgt_4,166,257). coords(fcgt_4,80,205,257).  visible(fcgt_4,257). happensAt(inactive(fcgt_4),257).
orientation(fcgt_2,157,258). coords(fcgt_2,64,152,258).  visible(fcgt_2,258). happensAt(walking(fcgt_2),258). orientation(fcgt_4,166,258). coords(fcgt_4,80,205,258).  visible(fcgt_4,258). happensAt(inactive(fcgt_4),258).
orientation(fcgt_2,157,259). coords(fcgt_2,65,151,259).  visible(fcgt_2,259). happensAt(walking(fcgt_2),259). orientation(fcgt_4,166,259). coords(fcgt_4,80,205,259).  visible(fcgt_4,259). happensAt(inactive(fcgt_4),259).
orientation(fcgt_2,157,260). coords(fcgt_2,66,151,260).  visible(fcgt_2,260). happensAt(walking(fcgt_2),260). orientation(fcgt_4,166,260). coords(fcgt_4,80,205,260).  visible(fcgt_4,260). happensAt(inactive(fcgt_4),260).
orientation(fcgt_2,157,261). coords(fcgt_2,67,150,261).  visible(fcgt_2,261). happensAt(walking(fcgt_2),261). orientation(fcgt_4,166,261). coords(fcgt_4,80,205,261).  visible(fcgt_4,261). happensAt(inactive(fcgt_4),261).
orientation(fcgt_2,157,262). coords(fcgt_2,68,150,262).  visible(fcgt_2,262). happensAt(walking(fcgt_2),262). orientation(fcgt_4,166,262). coords(fcgt_4,80,205,262).  visible(fcgt_4,262). happensAt(inactive(fcgt_4),262).
orientation(fcgt_2,157,263). coords(fcgt_2,68,149,263).  visible(fcgt_2,263). happensAt(walking(fcgt_2),263). orientation(fcgt_4,166,263). coords(fcgt_4,81,205,263).  visible(fcgt_4,263). happensAt(inactive(fcgt_4),263).
orientation(fcgt_2,157,264). coords(fcgt_2,70,148,264).  visible(fcgt_2,264). happensAt(walking(fcgt_2),264). orientation(fcgt_4,166,264). coords(fcgt_4,81,205,264).  visible(fcgt_4,264). happensAt(inactive(fcgt_4),264).
orientation(fcgt_2,157,265). coords(fcgt_2,70,147,265).  visible(fcgt_2,265). happensAt(walking(fcgt_2),265). orientation(fcgt_4,166,265). coords(fcgt_4,81,205,265).  visible(fcgt_4,265). happensAt(inactive(fcgt_4),265).
orientation(fcgt_2,157,266). coords(fcgt_2,71,146,266).  visible(fcgt_2,266). happensAt(walking(fcgt_2),266). orientation(fcgt_4,166,266). coords(fcgt_4,81,205,266).  visible(fcgt_4,266). happensAt(inactive(fcgt_4),266).
orientation(fcgt_2,157,267). coords(fcgt_2,71,146,267).  visible(fcgt_2,267). happensAt(walking(fcgt_2),267). orientation(fcgt_4,166,267). coords(fcgt_4,81,205,267).  visible(fcgt_4,267). happensAt(inactive(fcgt_4),267).
orientation(fcgt_2,157,268). coords(fcgt_2,71,145,268).  visible(fcgt_2,268). happensAt(walking(fcgt_2),268). orientation(fcgt_4,166,268). coords(fcgt_4,81,205,268).  visible(fcgt_4,268). happensAt(inactive(fcgt_4),268).
orientation(fcgt_2,157,269). coords(fcgt_2,71,145,269).  visible(fcgt_2,269). happensAt(walking(fcgt_2),269). orientation(fcgt_4,166,269). coords(fcgt_4,81,205,269).  visible(fcgt_4,269). happensAt(inactive(fcgt_4),269).
orientation(fcgt_2,157,270). coords(fcgt_2,71,144,270).  visible(fcgt_2,270). happensAt(walking(fcgt_2),270). orientation(fcgt_4,166,270). coords(fcgt_4,81,205,270).  visible(fcgt_4,270). happensAt(inactive(fcgt_4),270).
#show holdsAt/2.