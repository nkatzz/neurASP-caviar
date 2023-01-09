
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

	orientation(fomdgt3_1,150,120). coords(fomdgt3_1,130,110,120).  visible(fomdgt3_1,120). happensAt(walking(fomdgt3_1),120). orientation(fomdgt3_2,40,120). coords(fomdgt3_2,277,277,120).  visible(fomdgt3_2,120). happensAt(walking(fomdgt3_2),120).
orientation(fomdgt3_1,150,121). coords(fomdgt3_1,130,109,121).  visible(fomdgt3_1,121). happensAt(walking(fomdgt3_1),121). orientation(fomdgt3_2,40,121). coords(fomdgt3_2,275,272,121).  visible(fomdgt3_2,121). happensAt(walking(fomdgt3_2),121).
orientation(fomdgt3_1,150,122). coords(fomdgt3_1,130,109,122).  visible(fomdgt3_1,122). happensAt(walking(fomdgt3_1),122). orientation(fomdgt3_2,40,122). coords(fomdgt3_2,275,271,122).  visible(fomdgt3_2,122). happensAt(walking(fomdgt3_2),122).
orientation(fomdgt3_1,150,123). coords(fomdgt3_1,130,109,123).  visible(fomdgt3_1,123). happensAt(walking(fomdgt3_1),123). orientation(fomdgt3_2,40,123). coords(fomdgt3_2,275,271,123).  visible(fomdgt3_2,123). happensAt(walking(fomdgt3_2),123).
orientation(fomdgt3_1,150,124). coords(fomdgt3_1,130,109,124).  visible(fomdgt3_1,124). happensAt(walking(fomdgt3_1),124). orientation(fomdgt3_2,40,124). coords(fomdgt3_2,275,271,124).  visible(fomdgt3_2,124). happensAt(walking(fomdgt3_2),124).
orientation(fomdgt3_1,150,125). coords(fomdgt3_1,129,109,125).  visible(fomdgt3_1,125). happensAt(walking(fomdgt3_1),125). orientation(fomdgt3_2,50,125). coords(fomdgt3_2,275,269,125).  visible(fomdgt3_2,125). happensAt(walking(fomdgt3_2),125).
orientation(fomdgt3_1,150,126). coords(fomdgt3_1,129,109,126).  visible(fomdgt3_1,126). happensAt(walking(fomdgt3_1),126). orientation(fomdgt3_2,50,126). coords(fomdgt3_2,275,269,126).  visible(fomdgt3_2,126). happensAt(walking(fomdgt3_2),126).
orientation(fomdgt3_1,150,127). coords(fomdgt3_1,129,109,127).  visible(fomdgt3_1,127). happensAt(walking(fomdgt3_1),127). orientation(fomdgt3_2,50,127). coords(fomdgt3_2,275,269,127).  visible(fomdgt3_2,127). happensAt(walking(fomdgt3_2),127).
orientation(fomdgt3_1,150,128). coords(fomdgt3_1,126,108,128).  visible(fomdgt3_1,128). happensAt(walking(fomdgt3_1),128). orientation(fomdgt3_2,50,128). coords(fomdgt3_2,275,267,128).  visible(fomdgt3_2,128). happensAt(walking(fomdgt3_2),128).
orientation(fomdgt3_1,150,129). coords(fomdgt3_1,126,108,129).  visible(fomdgt3_1,129). happensAt(walking(fomdgt3_1),129). orientation(fomdgt3_2,50,129). coords(fomdgt3_2,275,267,129).  visible(fomdgt3_2,129). happensAt(walking(fomdgt3_2),129).
orientation(fomdgt3_1,150,130). coords(fomdgt3_1,124,108,130).  visible(fomdgt3_1,130). happensAt(walking(fomdgt3_1),130). orientation(fomdgt3_2,50,130). coords(fomdgt3_2,275,265,130).  visible(fomdgt3_2,130). happensAt(walking(fomdgt3_2),130).
orientation(fomdgt3_1,150,131). coords(fomdgt3_1,121,108,131).  visible(fomdgt3_1,131). happensAt(walking(fomdgt3_1),131). orientation(fomdgt3_2,50,131). coords(fomdgt3_2,275,263,131).  visible(fomdgt3_2,131). happensAt(walking(fomdgt3_2),131).
orientation(fomdgt3_1,150,132). coords(fomdgt3_1,121,108,132).  visible(fomdgt3_1,132). happensAt(walking(fomdgt3_1),132). orientation(fomdgt3_2,50,132). coords(fomdgt3_2,275,262,132).  visible(fomdgt3_2,132). happensAt(walking(fomdgt3_2),132).
orientation(fomdgt3_1,150,133). coords(fomdgt3_1,121,108,133).  visible(fomdgt3_1,133). happensAt(walking(fomdgt3_1),133). orientation(fomdgt3_2,50,133). coords(fomdgt3_2,275,262,133).  visible(fomdgt3_2,133). happensAt(walking(fomdgt3_2),133).
orientation(fomdgt3_1,150,134). coords(fomdgt3_1,120,108,134).  visible(fomdgt3_1,134). happensAt(walking(fomdgt3_1),134). orientation(fomdgt3_2,40,134). coords(fomdgt3_2,274,258,134).  visible(fomdgt3_2,134). happensAt(walking(fomdgt3_2),134).
orientation(fomdgt3_1,150,135). coords(fomdgt3_1,120,108,135).  visible(fomdgt3_1,135). happensAt(walking(fomdgt3_1),135). orientation(fomdgt3_2,40,135). coords(fomdgt3_2,274,256,135).  visible(fomdgt3_2,135). happensAt(walking(fomdgt3_2),135).
orientation(fomdgt3_1,150,136). coords(fomdgt3_1,120,108,136).  visible(fomdgt3_1,136). happensAt(walking(fomdgt3_1),136). orientation(fomdgt3_2,40,136). coords(fomdgt3_2,274,253,136).  visible(fomdgt3_2,136). happensAt(walking(fomdgt3_2),136).
orientation(fomdgt3_1,150,137). coords(fomdgt3_1,120,108,137).  visible(fomdgt3_1,137). happensAt(walking(fomdgt3_1),137). orientation(fomdgt3_2,40,137). coords(fomdgt3_2,273,249,137).  visible(fomdgt3_2,137). happensAt(walking(fomdgt3_2),137).
orientation(fomdgt3_1,150,138). coords(fomdgt3_1,118,108,138).  visible(fomdgt3_1,138). happensAt(walking(fomdgt3_1),138). orientation(fomdgt3_2,40,138). coords(fomdgt3_2,273,247,138).  visible(fomdgt3_2,138). happensAt(walking(fomdgt3_2),138).
orientation(fomdgt3_1,150,139). coords(fomdgt3_1,118,108,139).  visible(fomdgt3_1,139). happensAt(walking(fomdgt3_1),139). orientation(fomdgt3_2,40,139). coords(fomdgt3_2,272,246,139).  visible(fomdgt3_2,139). happensAt(walking(fomdgt3_2),139).
orientation(fomdgt3_1,150,140). coords(fomdgt3_1,118,108,140).  visible(fomdgt3_1,140). happensAt(walking(fomdgt3_1),140). orientation(fomdgt3_2,40,140). coords(fomdgt3_2,271,245,140).  visible(fomdgt3_2,140). happensAt(walking(fomdgt3_2),140).
orientation(fomdgt3_1,150,141). coords(fomdgt3_1,116,108,141).  visible(fomdgt3_1,141). happensAt(walking(fomdgt3_1),141). orientation(fomdgt3_2,40,141). coords(fomdgt3_2,271,244,141).  visible(fomdgt3_2,141). happensAt(walking(fomdgt3_2),141).
orientation(fomdgt3_1,150,142). coords(fomdgt3_1,116,108,142).  visible(fomdgt3_1,142). happensAt(walking(fomdgt3_1),142). orientation(fomdgt3_2,40,142). coords(fomdgt3_2,271,244,142).  visible(fomdgt3_2,142). happensAt(walking(fomdgt3_2),142).
orientation(fomdgt3_1,150,143). coords(fomdgt3_1,113,108,143).  visible(fomdgt3_1,143). happensAt(walking(fomdgt3_1),143). orientation(fomdgt3_2,40,143). coords(fomdgt3_2,271,244,143).  visible(fomdgt3_2,143). happensAt(walking(fomdgt3_2),143).
orientation(fomdgt3_1,150,144). coords(fomdgt3_1,113,108,144).  visible(fomdgt3_1,144). happensAt(walking(fomdgt3_1),144). orientation(fomdgt3_2,5,144). coords(fomdgt3_2,268,244,144).  visible(fomdgt3_2,144). happensAt(walking(fomdgt3_2),144).
#show holdsAt/2.