
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

	orientation(lb1gt_0,151,120). coords(lb1gt_0,180,120,120).  visible(lb1gt_0,120). happensAt(active(lb1gt_0),120). orientation(lb1gt_1,151,120). coords(lb1gt_1,84,124,120).  visible(lb1gt_1,120). happensAt(walking(lb1gt_1),120).
orientation(lb1gt_0,151,121). coords(lb1gt_0,180,120,121).  visible(lb1gt_0,121). happensAt(active(lb1gt_0),121). orientation(lb1gt_1,151,121). coords(lb1gt_1,84,124,121).  visible(lb1gt_1,121). happensAt(walking(lb1gt_1),121).
orientation(lb1gt_0,151,122). coords(lb1gt_0,180,120,122).  visible(lb1gt_0,122). happensAt(active(lb1gt_0),122). orientation(lb1gt_1,151,122). coords(lb1gt_1,84,125,122).  visible(lb1gt_1,122). happensAt(walking(lb1gt_1),122).
orientation(lb1gt_0,151,123). coords(lb1gt_0,180,119,123).  visible(lb1gt_0,123). happensAt(active(lb1gt_0),123). orientation(lb1gt_1,151,123). coords(lb1gt_1,84,126,123).  visible(lb1gt_1,123). happensAt(walking(lb1gt_1),123).
orientation(lb1gt_0,149,124). coords(lb1gt_0,180,119,124).  visible(lb1gt_0,124). happensAt(active(lb1gt_0),124). orientation(lb1gt_1,151,124). coords(lb1gt_1,84,127,124).  visible(lb1gt_1,124). happensAt(walking(lb1gt_1),124).
orientation(lb1gt_0,146,125). coords(lb1gt_0,180,119,125).  visible(lb1gt_0,125). happensAt(active(lb1gt_0),125). orientation(lb1gt_1,151,125). coords(lb1gt_1,84,129,125).  visible(lb1gt_1,125). happensAt(walking(lb1gt_1),125).
orientation(lb1gt_0,145,126). coords(lb1gt_0,180,117,126).  visible(lb1gt_0,126). happensAt(walking(lb1gt_0),126). orientation(lb1gt_1,151,126). coords(lb1gt_1,84,130,126).  visible(lb1gt_1,126). happensAt(walking(lb1gt_1),126).
orientation(lb1gt_0,145,127). coords(lb1gt_0,179,116,127).  visible(lb1gt_0,127). happensAt(walking(lb1gt_0),127). orientation(lb1gt_1,151,127). coords(lb1gt_1,85,131,127).  visible(lb1gt_1,127). happensAt(walking(lb1gt_1),127).
orientation(lb1gt_0,144,128). coords(lb1gt_0,179,115,128).  visible(lb1gt_0,128). happensAt(walking(lb1gt_0),128). orientation(lb1gt_1,151,128). coords(lb1gt_1,86,132,128).  visible(lb1gt_1,128). happensAt(walking(lb1gt_1),128).
orientation(lb1gt_0,143,129). coords(lb1gt_0,179,114,129).  visible(lb1gt_0,129). happensAt(walking(lb1gt_0),129). orientation(lb1gt_1,151,129). coords(lb1gt_1,86,132,129).  visible(lb1gt_1,129). happensAt(walking(lb1gt_1),129).
orientation(lb1gt_0,142,130). coords(lb1gt_0,179,113,130).  visible(lb1gt_0,130). happensAt(walking(lb1gt_0),130). orientation(lb1gt_1,151,130). coords(lb1gt_1,86,133,130).  visible(lb1gt_1,130). happensAt(walking(lb1gt_1),130).
orientation(lb1gt_0,142,131). coords(lb1gt_0,178,112,131).  visible(lb1gt_0,131). happensAt(walking(lb1gt_0),131). orientation(lb1gt_1,151,131). coords(lb1gt_1,86,133,131).  visible(lb1gt_1,131). happensAt(walking(lb1gt_1),131).
orientation(lb1gt_0,142,132). coords(lb1gt_0,177,111,132).  visible(lb1gt_0,132). happensAt(walking(lb1gt_0),132). orientation(lb1gt_1,151,132). coords(lb1gt_1,87,133,132).  visible(lb1gt_1,132). happensAt(walking(lb1gt_1),132).
orientation(lb1gt_0,142,133). coords(lb1gt_0,176,110,133).  visible(lb1gt_0,133). happensAt(walking(lb1gt_0),133). orientation(lb1gt_1,151,133). coords(lb1gt_1,87,133,133).  visible(lb1gt_1,133). happensAt(walking(lb1gt_1),133).
orientation(lb1gt_0,142,134). coords(lb1gt_0,175,109,134).  visible(lb1gt_0,134). happensAt(walking(lb1gt_0),134). orientation(lb1gt_1,151,134). coords(lb1gt_1,87,133,134).  visible(lb1gt_1,134). happensAt(walking(lb1gt_1),134).
orientation(lb1gt_0,142,135). coords(lb1gt_0,173,108,135).  visible(lb1gt_0,135). happensAt(walking(lb1gt_0),135). orientation(lb1gt_1,151,135). coords(lb1gt_1,87,134,135).  visible(lb1gt_1,135). happensAt(walking(lb1gt_1),135).
orientation(lb1gt_0,142,136). coords(lb1gt_0,173,107,136).  visible(lb1gt_0,136). happensAt(walking(lb1gt_0),136). orientation(lb1gt_1,151,136). coords(lb1gt_1,87,134,136).  visible(lb1gt_1,136). happensAt(walking(lb1gt_1),136).
orientation(lb1gt_0,142,137). coords(lb1gt_0,173,106,137).  visible(lb1gt_0,137). happensAt(walking(lb1gt_0),137). orientation(lb1gt_1,151,137). coords(lb1gt_1,87,134,137).  visible(lb1gt_1,137). happensAt(walking(lb1gt_1),137).
orientation(lb1gt_0,142,138). coords(lb1gt_0,173,104,138).  visible(lb1gt_0,138). happensAt(walking(lb1gt_0),138). orientation(lb1gt_1,151,138). coords(lb1gt_1,87,134,138).  visible(lb1gt_1,138). happensAt(walking(lb1gt_1),138).
orientation(lb1gt_0,138,139). coords(lb1gt_0,172,101,139).  visible(lb1gt_0,139). happensAt(walking(lb1gt_0),139). orientation(lb1gt_1,151,139). coords(lb1gt_1,87,134,139).  visible(lb1gt_1,139). happensAt(walking(lb1gt_1),139).
orientation(lb1gt_0,139,140). coords(lb1gt_0,172,98,140).  visible(lb1gt_0,140). happensAt(walking(lb1gt_0),140). orientation(lb1gt_1,151,140). coords(lb1gt_1,87,135,140).  visible(lb1gt_1,140). happensAt(walking(lb1gt_1),140).
orientation(lb1gt_0,138,141). coords(lb1gt_0,171,98,141).  visible(lb1gt_0,141). happensAt(walking(lb1gt_0),141). orientation(lb1gt_1,151,141). coords(lb1gt_1,88,135,141).  visible(lb1gt_1,141). happensAt(walking(lb1gt_1),141).
orientation(lb1gt_0,136,142). coords(lb1gt_0,171,96,142).  visible(lb1gt_0,142). happensAt(walking(lb1gt_0),142). orientation(lb1gt_1,151,142). coords(lb1gt_1,90,135,142).  visible(lb1gt_1,142). happensAt(walking(lb1gt_1),142).
orientation(lb1gt_0,136,143). coords(lb1gt_0,170,95,143).  visible(lb1gt_0,143). happensAt(walking(lb1gt_0),143). orientation(lb1gt_1,151,143). coords(lb1gt_1,90,135,143).  visible(lb1gt_1,143). happensAt(walking(lb1gt_1),143).
orientation(lb1gt_0,136,144). coords(lb1gt_0,170,93,144).  visible(lb1gt_0,144). happensAt(walking(lb1gt_0),144). orientation(lb1gt_1,151,144). coords(lb1gt_1,92,135,144).  visible(lb1gt_1,144). happensAt(walking(lb1gt_1),144).
#show holdsAt/2.