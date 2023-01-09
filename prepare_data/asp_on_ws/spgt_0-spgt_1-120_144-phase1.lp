
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

	orientation(spgt_0,152,120). coords(spgt_0,85,140,120).  visible(spgt_0,120). happensAt(walking(spgt_0),120). orientation(spgt_1,154,120). coords(spgt_1,84,127,120).  visible(spgt_1,120). happensAt(walking(spgt_1),120).
orientation(spgt_0,152,121). coords(spgt_0,86,140,121).  visible(spgt_0,121). happensAt(walking(spgt_0),121). orientation(spgt_1,154,121). coords(spgt_1,85,127,121).  visible(spgt_1,121). happensAt(walking(spgt_1),121).
orientation(spgt_0,152,122). coords(spgt_0,86,140,122).  visible(spgt_0,122). happensAt(walking(spgt_0),122). orientation(spgt_1,154,122). coords(spgt_1,86,126,122).  visible(spgt_1,122). happensAt(walking(spgt_1),122).
orientation(spgt_0,152,123). coords(spgt_0,86,139,123).  visible(spgt_0,123). happensAt(walking(spgt_0),123). orientation(spgt_1,154,123). coords(spgt_1,86,125,123).  visible(spgt_1,123). happensAt(walking(spgt_1),123).
orientation(spgt_0,152,124). coords(spgt_0,87,139,124).  visible(spgt_0,124). happensAt(walking(spgt_0),124). orientation(spgt_1,154,124). coords(spgt_1,87,124,124).  visible(spgt_1,124). happensAt(walking(spgt_1),124).
orientation(spgt_0,152,125). coords(spgt_0,87,138,125).  visible(spgt_0,125). happensAt(walking(spgt_0),125). orientation(spgt_1,154,125). coords(spgt_1,87,124,125).  visible(spgt_1,125). happensAt(walking(spgt_1),125).
orientation(spgt_0,152,126). coords(spgt_0,88,138,126).  visible(spgt_0,126). happensAt(walking(spgt_0),126). orientation(spgt_1,154,126). coords(spgt_1,87,124,126).  visible(spgt_1,126). happensAt(walking(spgt_1),126).
orientation(spgt_0,152,127). coords(spgt_0,89,138,127).  visible(spgt_0,127). happensAt(walking(spgt_0),127). orientation(spgt_1,154,127). coords(spgt_1,87,123,127).  visible(spgt_1,127). happensAt(walking(spgt_1),127).
orientation(spgt_0,152,128). coords(spgt_0,91,137,128).  visible(spgt_0,128). happensAt(walking(spgt_0),128). orientation(spgt_1,154,128). coords(spgt_1,89,123,128).  visible(spgt_1,128). happensAt(walking(spgt_1),128).
orientation(spgt_0,152,129). coords(spgt_0,92,137,129).  visible(spgt_0,129). happensAt(walking(spgt_0),129). orientation(spgt_1,154,129). coords(spgt_1,91,122,129).  visible(spgt_1,129). happensAt(walking(spgt_1),129).
orientation(spgt_0,152,130). coords(spgt_0,92,137,130).  visible(spgt_0,130). happensAt(walking(spgt_0),130). orientation(spgt_1,154,130). coords(spgt_1,91,122,130).  visible(spgt_1,130). happensAt(walking(spgt_1),130).
orientation(spgt_0,152,131). coords(spgt_0,93,137,131).  visible(spgt_0,131). happensAt(walking(spgt_0),131). orientation(spgt_1,154,131). coords(spgt_1,92,122,131).  visible(spgt_1,131). happensAt(walking(spgt_1),131).
orientation(spgt_0,152,132). coords(spgt_0,95,136,132).  visible(spgt_0,132). happensAt(walking(spgt_0),132). orientation(spgt_1,154,132). coords(spgt_1,93,121,132).  visible(spgt_1,132). happensAt(walking(spgt_1),132).
orientation(spgt_0,152,133). coords(spgt_0,95,136,133).  visible(spgt_0,133). happensAt(walking(spgt_0),133). orientation(spgt_1,154,133). coords(spgt_1,94,121,133).  visible(spgt_1,133). happensAt(walking(spgt_1),133).
orientation(spgt_0,148,134). coords(spgt_0,95,135,134).  visible(spgt_0,134). happensAt(walking(spgt_0),134). orientation(spgt_1,154,134). coords(spgt_1,94,120,134).  visible(spgt_1,134). happensAt(walking(spgt_1),134).
orientation(spgt_0,148,135). coords(spgt_0,96,135,135).  visible(spgt_0,135). happensAt(walking(spgt_0),135). orientation(spgt_1,154,135). coords(spgt_1,94,120,135).  visible(spgt_1,135). happensAt(walking(spgt_1),135).
orientation(spgt_0,148,136). coords(spgt_0,96,135,136).  visible(spgt_0,136). happensAt(walking(spgt_0),136). orientation(spgt_1,154,136). coords(spgt_1,94,119,136).  visible(spgt_1,136). happensAt(walking(spgt_1),136).
orientation(spgt_0,152,137). coords(spgt_0,96,133,137).  visible(spgt_0,137). happensAt(walking(spgt_0),137). orientation(spgt_1,154,137). coords(spgt_1,95,118,137).  visible(spgt_1,137). happensAt(walking(spgt_1),137).
orientation(spgt_0,152,138). coords(spgt_0,96,132,138).  visible(spgt_0,138). happensAt(walking(spgt_0),138). orientation(spgt_1,154,138). coords(spgt_1,96,117,138).  visible(spgt_1,138). happensAt(walking(spgt_1),138).
orientation(spgt_0,153,139). coords(spgt_0,96,131,139).  visible(spgt_0,139). happensAt(walking(spgt_0),139). orientation(spgt_1,154,139). coords(spgt_1,96,117,139).  visible(spgt_1,139). happensAt(walking(spgt_1),139).
orientation(spgt_0,151,140). coords(spgt_0,97,130,140).  visible(spgt_0,140). happensAt(walking(spgt_0),140). orientation(spgt_1,148,140). coords(spgt_1,96,116,140).  visible(spgt_1,140). happensAt(walking(spgt_1),140).
orientation(spgt_0,154,141). coords(spgt_0,99,129,141).  visible(spgt_0,141). happensAt(walking(spgt_0),141). orientation(spgt_1,148,141). coords(spgt_1,96,116,141).  visible(spgt_1,141). happensAt(walking(spgt_1),141).
orientation(spgt_0,155,142). coords(spgt_0,101,129,142).  visible(spgt_0,142). happensAt(walking(spgt_0),142). orientation(spgt_1,148,142). coords(spgt_1,97,114,142).  visible(spgt_1,142). happensAt(walking(spgt_1),142).
orientation(spgt_0,155,143). coords(spgt_0,101,128,143).  visible(spgt_0,143). happensAt(walking(spgt_0),143). orientation(spgt_1,148,143). coords(spgt_1,98,114,143).  visible(spgt_1,143). happensAt(walking(spgt_1),143).
orientation(spgt_0,155,144). coords(spgt_0,103,128,144).  visible(spgt_0,144). happensAt(walking(spgt_0),144). orientation(spgt_1,148,144). coords(spgt_1,99,113,144).  visible(spgt_1,144). happensAt(walking(spgt_1),144).
#show holdsAt/2.