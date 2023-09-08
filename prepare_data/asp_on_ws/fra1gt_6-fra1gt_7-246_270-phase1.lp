
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

	orientation(fra1gt_6,123,246). coords(fra1gt_6,240,35,246).  visible(fra1gt_6,246). happensAt(walking(fra1gt_6),246). orientation(fra1gt_7,137,246). coords(fra1gt_7,151,37,246).  visible(fra1gt_7,246). happensAt(walking(fra1gt_7),246).
orientation(fra1gt_6,123,247). coords(fra1gt_6,238,36,247).  visible(fra1gt_6,247). happensAt(walking(fra1gt_6),247). orientation(fra1gt_7,137,247). coords(fra1gt_7,151,37,247).  visible(fra1gt_7,247). happensAt(walking(fra1gt_7),247).
orientation(fra1gt_6,123,248). coords(fra1gt_6,236,37,248).  visible(fra1gt_6,248). happensAt(walking(fra1gt_6),248). orientation(fra1gt_7,137,248). coords(fra1gt_7,151,37,248).  visible(fra1gt_7,248). happensAt(walking(fra1gt_7),248).
orientation(fra1gt_6,123,249). coords(fra1gt_6,235,40,249).  visible(fra1gt_6,249). happensAt(walking(fra1gt_6),249). orientation(fra1gt_7,137,249). coords(fra1gt_7,151,38,249).  visible(fra1gt_7,249). happensAt(walking(fra1gt_7),249).
orientation(fra1gt_6,123,250). coords(fra1gt_6,234,43,250).  visible(fra1gt_6,250). happensAt(walking(fra1gt_6),250). orientation(fra1gt_7,137,250). coords(fra1gt_7,151,38,250).  visible(fra1gt_7,250). happensAt(walking(fra1gt_7),250).
orientation(fra1gt_6,123,251). coords(fra1gt_6,233,45,251).  visible(fra1gt_6,251). happensAt(walking(fra1gt_6),251). orientation(fra1gt_7,137,251). coords(fra1gt_7,150,39,251).  visible(fra1gt_7,251). happensAt(walking(fra1gt_7),251).
orientation(fra1gt_6,123,252). coords(fra1gt_6,233,46,252).  visible(fra1gt_6,252). happensAt(walking(fra1gt_6),252). orientation(fra1gt_7,137,252). coords(fra1gt_7,150,39,252).  visible(fra1gt_7,252). happensAt(walking(fra1gt_7),252).
orientation(fra1gt_6,123,253). coords(fra1gt_6,232,48,253).  visible(fra1gt_6,253). happensAt(walking(fra1gt_6),253). orientation(fra1gt_7,137,253). coords(fra1gt_7,150,39,253).  visible(fra1gt_7,253). happensAt(walking(fra1gt_7),253).
orientation(fra1gt_6,123,254). coords(fra1gt_6,232,49,254).  visible(fra1gt_6,254). happensAt(walking(fra1gt_6),254). orientation(fra1gt_7,137,254). coords(fra1gt_7,149,40,254).  visible(fra1gt_7,254). happensAt(walking(fra1gt_7),254).
orientation(fra1gt_6,123,255). coords(fra1gt_6,232,49,255).  visible(fra1gt_6,255). happensAt(walking(fra1gt_6),255). orientation(fra1gt_7,137,255). coords(fra1gt_7,149,40,255).  visible(fra1gt_7,255). happensAt(walking(fra1gt_7),255).
orientation(fra1gt_6,123,256). coords(fra1gt_6,231,50,256).  visible(fra1gt_6,256). happensAt(walking(fra1gt_6),256). orientation(fra1gt_7,137,256). coords(fra1gt_7,149,40,256).  visible(fra1gt_7,256). happensAt(walking(fra1gt_7),256).
orientation(fra1gt_6,123,257). coords(fra1gt_6,230,51,257).  visible(fra1gt_6,257). happensAt(walking(fra1gt_6),257). orientation(fra1gt_7,137,257). coords(fra1gt_7,148,41,257).  visible(fra1gt_7,257). happensAt(walking(fra1gt_7),257).
orientation(fra1gt_6,123,258). coords(fra1gt_6,228,51,258).  visible(fra1gt_6,258). happensAt(walking(fra1gt_6),258). orientation(fra1gt_7,137,258). coords(fra1gt_7,148,41,258).  visible(fra1gt_7,258). happensAt(walking(fra1gt_7),258).
orientation(fra1gt_6,123,259). coords(fra1gt_6,227,52,259).  visible(fra1gt_6,259). happensAt(walking(fra1gt_6),259). orientation(fra1gt_7,137,259). coords(fra1gt_7,147,42,259).  visible(fra1gt_7,259). happensAt(walking(fra1gt_7),259).
orientation(fra1gt_6,123,260). coords(fra1gt_6,226,52,260).  visible(fra1gt_6,260). happensAt(walking(fra1gt_6),260). orientation(fra1gt_7,137,260). coords(fra1gt_7,147,42,260).  visible(fra1gt_7,260). happensAt(walking(fra1gt_7),260).
orientation(fra1gt_6,123,261). coords(fra1gt_6,224,52,261).  visible(fra1gt_6,261). happensAt(walking(fra1gt_6),261). orientation(fra1gt_7,137,261). coords(fra1gt_7,146,42,261).  visible(fra1gt_7,261). happensAt(walking(fra1gt_7),261).
orientation(fra1gt_6,123,262). coords(fra1gt_6,223,53,262).  visible(fra1gt_6,262). happensAt(walking(fra1gt_6),262). orientation(fra1gt_7,137,262). coords(fra1gt_7,146,42,262).  visible(fra1gt_7,262). happensAt(walking(fra1gt_7),262).
orientation(fra1gt_6,123,263). coords(fra1gt_6,222,53,263).  visible(fra1gt_6,263). happensAt(walking(fra1gt_6),263). orientation(fra1gt_7,137,263). coords(fra1gt_7,144,43,263).  visible(fra1gt_7,263). happensAt(walking(fra1gt_7),263).
orientation(fra1gt_6,123,264). coords(fra1gt_6,221,54,264).  visible(fra1gt_6,264). happensAt(walking(fra1gt_6),264). orientation(fra1gt_7,137,264). coords(fra1gt_7,144,43,264).  visible(fra1gt_7,264). happensAt(walking(fra1gt_7),264).
orientation(fra1gt_6,123,265). coords(fra1gt_6,221,54,265).  visible(fra1gt_6,265). happensAt(walking(fra1gt_6),265). orientation(fra1gt_7,137,265). coords(fra1gt_7,143,43,265).  visible(fra1gt_7,265). happensAt(walking(fra1gt_7),265).
orientation(fra1gt_6,123,266). coords(fra1gt_6,220,56,266).  visible(fra1gt_6,266). happensAt(walking(fra1gt_6),266). orientation(fra1gt_7,137,266). coords(fra1gt_7,143,43,266).  visible(fra1gt_7,266). happensAt(walking(fra1gt_7),266).
orientation(fra1gt_6,123,267). coords(fra1gt_6,220,57,267).  visible(fra1gt_6,267). happensAt(walking(fra1gt_6),267). orientation(fra1gt_7,137,267). coords(fra1gt_7,143,44,267).  visible(fra1gt_7,267). happensAt(walking(fra1gt_7),267).
orientation(fra1gt_6,123,268). coords(fra1gt_6,219,58,268).  visible(fra1gt_6,268). happensAt(walking(fra1gt_6),268). orientation(fra1gt_7,137,268). coords(fra1gt_7,143,45,268).  visible(fra1gt_7,268). happensAt(walking(fra1gt_7),268).
orientation(fra1gt_6,123,269). coords(fra1gt_6,219,59,269).  visible(fra1gt_6,269). happensAt(walking(fra1gt_6),269). orientation(fra1gt_7,137,269). coords(fra1gt_7,143,45,269).  visible(fra1gt_7,269). happensAt(walking(fra1gt_7),269).
orientation(fra1gt_6,123,270). coords(fra1gt_6,218,60,270).  visible(fra1gt_6,270). happensAt(walking(fra1gt_6),270). orientation(fra1gt_7,137,270). coords(fra1gt_7,142,45,270).  visible(fra1gt_7,270). happensAt(walking(fra1gt_7),270).
#show holdsAt/2.