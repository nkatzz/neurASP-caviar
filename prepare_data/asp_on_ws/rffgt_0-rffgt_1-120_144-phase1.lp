
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

	orientation(rffgt_0,135,120). coords(rffgt_0,148,39,120).  visible(rffgt_0,120). happensAt(walking(rffgt_0),120). orientation(rffgt_1,146,120). coords(rffgt_1,152,110,120).  visible(rffgt_1,120). happensAt(inactive(rffgt_1),120).
orientation(rffgt_0,135,121). coords(rffgt_0,148,39,121).  visible(rffgt_0,121). happensAt(walking(rffgt_0),121). orientation(rffgt_1,146,121). coords(rffgt_1,152,109,121).  visible(rffgt_1,121). happensAt(inactive(rffgt_1),121).
orientation(rffgt_0,135,122). coords(rffgt_0,149,39,122).  visible(rffgt_0,122). happensAt(walking(rffgt_0),122). orientation(rffgt_1,146,122). coords(rffgt_1,152,109,122).  visible(rffgt_1,122). happensAt(inactive(rffgt_1),122).
orientation(rffgt_0,135,123). coords(rffgt_0,150,39,123).  visible(rffgt_0,123). happensAt(walking(rffgt_0),123). orientation(rffgt_1,148,123). coords(rffgt_1,152,109,123).  visible(rffgt_1,123). happensAt(inactive(rffgt_1),123).
orientation(rffgt_0,135,124). coords(rffgt_0,151,39,124).  visible(rffgt_0,124). happensAt(walking(rffgt_0),124). orientation(rffgt_1,148,124). coords(rffgt_1,152,109,124).  visible(rffgt_1,124). happensAt(inactive(rffgt_1),124).
orientation(rffgt_0,135,125). coords(rffgt_0,151,38,125).  visible(rffgt_0,125). happensAt(walking(rffgt_0),125). orientation(rffgt_1,148,125). coords(rffgt_1,152,109,125).  visible(rffgt_1,125). happensAt(inactive(rffgt_1),125).
orientation(rffgt_0,135,126). coords(rffgt_0,150,38,126).  visible(rffgt_0,126). happensAt(walking(rffgt_0),126). orientation(rffgt_1,148,126). coords(rffgt_1,152,109,126).  visible(rffgt_1,126). happensAt(inactive(rffgt_1),126).
orientation(rffgt_0,135,127). coords(rffgt_0,151,38,127).  visible(rffgt_0,127). happensAt(walking(rffgt_0),127). orientation(rffgt_1,148,127). coords(rffgt_1,152,109,127).  visible(rffgt_1,127). happensAt(inactive(rffgt_1),127).
orientation(rffgt_0,135,128). coords(rffgt_0,152,37,128).  visible(rffgt_0,128). happensAt(walking(rffgt_0),128). orientation(rffgt_1,148,128). coords(rffgt_1,152,109,128).  visible(rffgt_1,128). happensAt(inactive(rffgt_1),128).
orientation(rffgt_0,135,129). coords(rffgt_0,152,37,129).  visible(rffgt_0,129). happensAt(walking(rffgt_0),129). orientation(rffgt_1,148,129). coords(rffgt_1,152,109,129).  visible(rffgt_1,129). happensAt(inactive(rffgt_1),129).
orientation(rffgt_0,135,130). coords(rffgt_0,152,37,130).  visible(rffgt_0,130). happensAt(walking(rffgt_0),130). orientation(rffgt_1,148,130). coords(rffgt_1,152,109,130).  visible(rffgt_1,130). happensAt(inactive(rffgt_1),130).
orientation(rffgt_0,135,131). coords(rffgt_0,152,37,131).  visible(rffgt_0,131). happensAt(walking(rffgt_0),131). orientation(rffgt_1,148,131). coords(rffgt_1,152,109,131).  visible(rffgt_1,131). happensAt(inactive(rffgt_1),131).
orientation(rffgt_0,135,132). coords(rffgt_0,153,34,132).  visible(rffgt_0,132). happensAt(walking(rffgt_0),132). orientation(rffgt_1,148,132). coords(rffgt_1,152,109,132).  visible(rffgt_1,132). happensAt(inactive(rffgt_1),132).
orientation(rffgt_0,135,133). coords(rffgt_0,153,34,133).  visible(rffgt_0,133). happensAt(walking(rffgt_0),133). orientation(rffgt_1,148,133). coords(rffgt_1,152,109,133).  visible(rffgt_1,133). happensAt(inactive(rffgt_1),133).
orientation(rffgt_0,135,134). coords(rffgt_0,154,34,134).  visible(rffgt_0,134). happensAt(walking(rffgt_0),134). orientation(rffgt_1,148,134). coords(rffgt_1,152,109,134).  visible(rffgt_1,134). happensAt(inactive(rffgt_1),134).
orientation(rffgt_0,135,135). coords(rffgt_0,155,34,135).  visible(rffgt_0,135). happensAt(walking(rffgt_0),135). orientation(rffgt_1,148,135). coords(rffgt_1,152,109,135).  visible(rffgt_1,135). happensAt(inactive(rffgt_1),135).
orientation(rffgt_0,135,136). coords(rffgt_0,155,34,136).  visible(rffgt_0,136). happensAt(walking(rffgt_0),136). orientation(rffgt_1,148,136). coords(rffgt_1,152,109,136).  visible(rffgt_1,136). happensAt(inactive(rffgt_1),136).
orientation(rffgt_0,135,137). coords(rffgt_0,157,34,137).  visible(rffgt_0,137). happensAt(walking(rffgt_0),137). orientation(rffgt_1,148,137). coords(rffgt_1,152,109,137).  visible(rffgt_1,137). happensAt(inactive(rffgt_1),137).
orientation(rffgt_0,135,138). coords(rffgt_0,157,34,138).  visible(rffgt_0,138). happensAt(walking(rffgt_0),138). orientation(rffgt_1,148,138). coords(rffgt_1,152,109,138).  visible(rffgt_1,138). happensAt(inactive(rffgt_1),138).
orientation(rffgt_0,135,139). coords(rffgt_0,157,33,139).  visible(rffgt_0,139). happensAt(walking(rffgt_0),139). orientation(rffgt_1,148,139). coords(rffgt_1,152,109,139).  visible(rffgt_1,139). happensAt(inactive(rffgt_1),139).
orientation(rffgt_0,135,140). coords(rffgt_0,157,33,140).  visible(rffgt_0,140). happensAt(walking(rffgt_0),140). orientation(rffgt_1,148,140). coords(rffgt_1,152,109,140).  visible(rffgt_1,140). happensAt(inactive(rffgt_1),140).
orientation(rffgt_0,135,141). coords(rffgt_0,157,32,141).  visible(rffgt_0,141). happensAt(walking(rffgt_0),141). orientation(rffgt_1,148,141). coords(rffgt_1,152,109,141).  visible(rffgt_1,141). happensAt(inactive(rffgt_1),141).
orientation(rffgt_0,135,142). coords(rffgt_0,158,32,142).  visible(rffgt_0,142). happensAt(walking(rffgt_0),142). orientation(rffgt_1,148,142). coords(rffgt_1,152,109,142).  visible(rffgt_1,142). happensAt(inactive(rffgt_1),142).
orientation(rffgt_0,135,143). coords(rffgt_0,158,32,143).  visible(rffgt_0,143). happensAt(walking(rffgt_0),143). orientation(rffgt_1,148,143). coords(rffgt_1,152,109,143).  visible(rffgt_1,143). happensAt(inactive(rffgt_1),143).
orientation(rffgt_0,135,144). coords(rffgt_0,158,32,144).  visible(rffgt_0,144). happensAt(walking(rffgt_0),144). orientation(rffgt_1,148,144). coords(rffgt_1,152,109,144).  visible(rffgt_1,144). happensAt(inactive(rffgt_1),144).
#show holdsAt/2.