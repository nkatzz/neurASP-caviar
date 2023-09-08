
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

	orientation(fra2gt_0,9,126). coords(fra2gt_0,249,232,126).  visible(fra2gt_0,126). happensAt(walking(fra2gt_0),126). orientation(fra2gt_5,150,126). coords(fra2gt_5,26,131,126).  visible(fra2gt_5,126). happensAt(walking(fra2gt_5),126).
orientation(fra2gt_0,10,127). coords(fra2gt_0,251,236,127).  visible(fra2gt_0,127). happensAt(walking(fra2gt_0),127). orientation(fra2gt_5,150,127). coords(fra2gt_5,26,131,127).  visible(fra2gt_5,127). happensAt(walking(fra2gt_5),127).
orientation(fra2gt_0,18,128). coords(fra2gt_0,253,241,128).  visible(fra2gt_0,128). happensAt(walking(fra2gt_0),128). orientation(fra2gt_5,150,128). coords(fra2gt_5,26,131,128).  visible(fra2gt_5,128). happensAt(walking(fra2gt_5),128).
orientation(fra2gt_0,28,129). coords(fra2gt_0,257,246,129).  visible(fra2gt_0,129). happensAt(walking(fra2gt_0),129). orientation(fra2gt_5,150,129). coords(fra2gt_5,26,131,129).  visible(fra2gt_5,129). happensAt(walking(fra2gt_5),129).
orientation(fra2gt_0,28,130). coords(fra2gt_0,259,250,130).  visible(fra2gt_0,130). happensAt(walking(fra2gt_0),130). orientation(fra2gt_5,150,130). coords(fra2gt_5,26,131,130).  visible(fra2gt_5,130). happensAt(walking(fra2gt_5),130).
orientation(fra2gt_0,28,131). coords(fra2gt_0,263,251,131).  visible(fra2gt_0,131). happensAt(walking(fra2gt_0),131). orientation(fra2gt_5,150,131). coords(fra2gt_5,26,131,131).  visible(fra2gt_5,131). happensAt(walking(fra2gt_5),131).
orientation(fra2gt_0,28,132). coords(fra2gt_0,265,254,132).  visible(fra2gt_0,132). happensAt(walking(fra2gt_0),132). orientation(fra2gt_5,150,132). coords(fra2gt_5,26,131,132).  visible(fra2gt_5,132). happensAt(walking(fra2gt_5),132).
orientation(fra2gt_0,28,133). coords(fra2gt_0,267,257,133).  visible(fra2gt_0,133). happensAt(walking(fra2gt_0),133). orientation(fra2gt_5,150,133). coords(fra2gt_5,26,131,133).  visible(fra2gt_5,133). happensAt(walking(fra2gt_5),133).
orientation(fra2gt_0,28,134). coords(fra2gt_0,270,261,134).  visible(fra2gt_0,134). happensAt(walking(fra2gt_0),134). orientation(fra2gt_5,150,134). coords(fra2gt_5,26,130,134).  visible(fra2gt_5,134). happensAt(walking(fra2gt_5),134).
orientation(fra2gt_0,28,135). coords(fra2gt_0,271,264,135).  visible(fra2gt_0,135). happensAt(walking(fra2gt_0),135). orientation(fra2gt_5,150,135). coords(fra2gt_5,26,130,135).  visible(fra2gt_5,135). happensAt(walking(fra2gt_5),135).
orientation(fra2gt_0,39,136). coords(fra2gt_0,273,266,136).  visible(fra2gt_0,136). happensAt(walking(fra2gt_0),136). orientation(fra2gt_5,150,136). coords(fra2gt_5,26,128,136).  visible(fra2gt_5,136). happensAt(walking(fra2gt_5),136).
orientation(fra2gt_0,47,137). coords(fra2gt_0,274,269,137).  visible(fra2gt_0,137). happensAt(walking(fra2gt_0),137). orientation(fra2gt_5,150,137). coords(fra2gt_5,26,128,137).  visible(fra2gt_5,137). happensAt(walking(fra2gt_5),137).
orientation(fra2gt_0,51,138). coords(fra2gt_0,274,271,138).  visible(fra2gt_0,138). happensAt(walking(fra2gt_0),138). orientation(fra2gt_5,150,138). coords(fra2gt_5,26,125,138).  visible(fra2gt_5,138). happensAt(walking(fra2gt_5),138).
orientation(fra2gt_0,51,139). coords(fra2gt_0,275,273,139).  visible(fra2gt_0,139). happensAt(walking(fra2gt_0),139). orientation(fra2gt_5,150,139). coords(fra2gt_5,27,125,139).  visible(fra2gt_5,139). happensAt(walking(fra2gt_5),139).
orientation(fra2gt_0,51,140). coords(fra2gt_0,276,273,140).  visible(fra2gt_0,140). happensAt(walking(fra2gt_0),140). orientation(fra2gt_5,150,140). coords(fra2gt_5,27,125,140).  visible(fra2gt_5,140). happensAt(walking(fra2gt_5),140).
orientation(fra2gt_0,51,141). coords(fra2gt_0,278,273,141).  visible(fra2gt_0,141). happensAt(walking(fra2gt_0),141). orientation(fra2gt_5,150,141). coords(fra2gt_5,28,124,141).  visible(fra2gt_5,141). happensAt(walking(fra2gt_5),141).
orientation(fra2gt_0,69,142). coords(fra2gt_0,282,273,142).  visible(fra2gt_0,142). happensAt(walking(fra2gt_0),142). orientation(fra2gt_5,150,142). coords(fra2gt_5,28,124,142).  visible(fra2gt_5,142). happensAt(walking(fra2gt_5),142).
orientation(fra2gt_0,77,143). coords(fra2gt_0,284,274,143).  visible(fra2gt_0,143). happensAt(walking(fra2gt_0),143). orientation(fra2gt_5,150,143). coords(fra2gt_5,28,124,143).  visible(fra2gt_5,143). happensAt(walking(fra2gt_5),143).
orientation(fra2gt_0,77,144). coords(fra2gt_0,289,275,144).  visible(fra2gt_0,144). happensAt(walking(fra2gt_0),144). orientation(fra2gt_5,150,144). coords(fra2gt_5,28,123,144).  visible(fra2gt_5,144). happensAt(walking(fra2gt_5),144).
orientation(fra2gt_0,77,145). coords(fra2gt_0,291,276,145).  visible(fra2gt_0,145). happensAt(walking(fra2gt_0),145). orientation(fra2gt_5,150,145). coords(fra2gt_5,28,123,145).  visible(fra2gt_5,145). happensAt(walking(fra2gt_5),145).
orientation(fra2gt_0,77,146). coords(fra2gt_0,294,277,146).  visible(fra2gt_0,146). happensAt(walking(fra2gt_0),146). orientation(fra2gt_5,150,146). coords(fra2gt_5,28,123,146).  visible(fra2gt_5,146). happensAt(walking(fra2gt_5),146).
orientation(fra2gt_0,77,147). coords(fra2gt_0,295,279,147).  visible(fra2gt_0,147). happensAt(walking(fra2gt_0),147). orientation(fra2gt_5,150,147). coords(fra2gt_5,28,121,147).  visible(fra2gt_5,147). happensAt(walking(fra2gt_5),147).
orientation(fra2gt_0,77,148). coords(fra2gt_0,297,281,148).  visible(fra2gt_0,148). happensAt(walking(fra2gt_0),148). orientation(fra2gt_5,150,148). coords(fra2gt_5,28,121,148).  visible(fra2gt_5,148). happensAt(walking(fra2gt_5),148).
orientation(fra2gt_0,77,149). coords(fra2gt_0,300,283,149).  visible(fra2gt_0,149). happensAt(walking(fra2gt_0),149). orientation(fra2gt_5,150,149). coords(fra2gt_5,29,120,149).  visible(fra2gt_5,149). happensAt(walking(fra2gt_5),149).
orientation(fra2gt_0,77,150). coords(fra2gt_0,300,286,150).  happensAt(disappear(fra2gt_0),150). happensAt(walking(fra2gt_0),150). orientation(fra2gt_5,150,150). coords(fra2gt_5,29,120,150).  visible(fra2gt_5,150). happensAt(walking(fra2gt_5),150).
#show holdsAt/2.