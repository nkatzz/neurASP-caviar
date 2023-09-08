
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

	orientation(mws1gt_1,155,114). coords(mws1gt_1,102,171,114).  visible(mws1gt_1,114). happensAt(walking(mws1gt_1),114). orientation(mws1gt_4,0,114). coords(mws1gt_4,26,153,114).  visible(mws1gt_4,114). happensAt(walking(mws1gt_4),114).
orientation(mws1gt_1,155,115). coords(mws1gt_1,102,170,115).  visible(mws1gt_1,115). happensAt(walking(mws1gt_1),115). orientation(mws1gt_4,0,115). coords(mws1gt_4,26,152,115).  visible(mws1gt_4,115). happensAt(walking(mws1gt_4),115).
orientation(mws1gt_1,155,116). coords(mws1gt_1,103,169,116).  visible(mws1gt_1,116). happensAt(walking(mws1gt_1),116). orientation(mws1gt_4,0,116). coords(mws1gt_4,26,151,116).  visible(mws1gt_4,116). happensAt(walking(mws1gt_4),116).
orientation(mws1gt_1,155,117). coords(mws1gt_1,105,169,117).  visible(mws1gt_1,117). happensAt(walking(mws1gt_1),117). orientation(mws1gt_4,0,117). coords(mws1gt_4,26,151,117).  visible(mws1gt_4,117). happensAt(walking(mws1gt_4),117).
orientation(mws1gt_1,155,118). coords(mws1gt_1,107,169,118).  visible(mws1gt_1,118). happensAt(walking(mws1gt_1),118). orientation(mws1gt_4,0,118). coords(mws1gt_4,27,150,118).  visible(mws1gt_4,118). happensAt(walking(mws1gt_4),118).
orientation(mws1gt_1,155,119). coords(mws1gt_1,108,168,119).  visible(mws1gt_1,119). happensAt(walking(mws1gt_1),119). orientation(mws1gt_4,0,119). coords(mws1gt_4,27,149,119).  visible(mws1gt_4,119). happensAt(walking(mws1gt_4),119).
orientation(mws1gt_1,155,120). coords(mws1gt_1,110,168,120).  visible(mws1gt_1,120). happensAt(walking(mws1gt_1),120). orientation(mws1gt_4,0,120). coords(mws1gt_4,27,148,120).  visible(mws1gt_4,120). happensAt(walking(mws1gt_4),120).
orientation(mws1gt_1,155,121). coords(mws1gt_1,111,167,121).  visible(mws1gt_1,121). happensAt(walking(mws1gt_1),121). orientation(mws1gt_4,0,121). coords(mws1gt_4,27,148,121).  visible(mws1gt_4,121). happensAt(walking(mws1gt_4),121).
orientation(mws1gt_1,155,122). coords(mws1gt_1,112,166,122).  visible(mws1gt_1,122). happensAt(walking(mws1gt_1),122). orientation(mws1gt_4,0,122). coords(mws1gt_4,28,147,122).  visible(mws1gt_4,122). happensAt(walking(mws1gt_4),122).
orientation(mws1gt_1,155,123). coords(mws1gt_1,113,165,123).  visible(mws1gt_1,123). happensAt(walking(mws1gt_1),123). orientation(mws1gt_4,0,123). coords(mws1gt_4,27,146,123).  visible(mws1gt_4,123). happensAt(walking(mws1gt_4),123).
orientation(mws1gt_1,155,124). coords(mws1gt_1,113,164,124).  visible(mws1gt_1,124). happensAt(walking(mws1gt_1),124). orientation(mws1gt_4,0,124). coords(mws1gt_4,27,145,124).  visible(mws1gt_4,124). happensAt(walking(mws1gt_4),124).
orientation(mws1gt_1,155,125). coords(mws1gt_1,114,163,125).  visible(mws1gt_1,125). happensAt(walking(mws1gt_1),125). orientation(mws1gt_4,0,125). coords(mws1gt_4,27,144,125).  visible(mws1gt_4,125). happensAt(walking(mws1gt_4),125).
orientation(mws1gt_1,155,126). coords(mws1gt_1,115,162,126).  visible(mws1gt_1,126). happensAt(walking(mws1gt_1),126). orientation(mws1gt_4,0,126). coords(mws1gt_4,27,144,126).  visible(mws1gt_4,126). happensAt(walking(mws1gt_4),126).
orientation(mws1gt_1,155,127). coords(mws1gt_1,115,162,127).  visible(mws1gt_1,127). happensAt(walking(mws1gt_1),127). orientation(mws1gt_4,0,127). coords(mws1gt_4,27,142,127).  visible(mws1gt_4,127). happensAt(walking(mws1gt_4),127).
orientation(mws1gt_1,155,128). coords(mws1gt_1,115,161,128).  visible(mws1gt_1,128). happensAt(walking(mws1gt_1),128). orientation(mws1gt_4,0,128). coords(mws1gt_4,27,142,128).  visible(mws1gt_4,128). happensAt(walking(mws1gt_4),128).
orientation(mws1gt_1,155,129). coords(mws1gt_1,115,159,129).  visible(mws1gt_1,129). happensAt(walking(mws1gt_1),129). orientation(mws1gt_4,0,129). coords(mws1gt_4,27,142,129).  visible(mws1gt_4,129). happensAt(walking(mws1gt_4),129).
orientation(mws1gt_1,155,130). coords(mws1gt_1,115,157,130).  visible(mws1gt_1,130). happensAt(walking(mws1gt_1),130). orientation(mws1gt_4,0,130). coords(mws1gt_4,27,141,130).  visible(mws1gt_4,130). happensAt(walking(mws1gt_4),130).
orientation(mws1gt_1,155,131). coords(mws1gt_1,118,156,131).  visible(mws1gt_1,131). happensAt(walking(mws1gt_1),131). orientation(mws1gt_4,0,131). coords(mws1gt_4,27,141,131).  visible(mws1gt_4,131). happensAt(walking(mws1gt_4),131).
orientation(mws1gt_1,155,132). coords(mws1gt_1,120,156,132).  visible(mws1gt_1,132). happensAt(walking(mws1gt_1),132). orientation(mws1gt_4,0,132). coords(mws1gt_4,27,141,132).  visible(mws1gt_4,132). happensAt(walking(mws1gt_4),132).
orientation(mws1gt_1,155,133). coords(mws1gt_1,122,155,133).  visible(mws1gt_1,133). happensAt(walking(mws1gt_1),133). orientation(mws1gt_4,0,133). coords(mws1gt_4,27,141,133).  visible(mws1gt_4,133). happensAt(walking(mws1gt_4),133).
orientation(mws1gt_1,155,134). coords(mws1gt_1,125,155,134).  visible(mws1gt_1,134). happensAt(walking(mws1gt_1),134). orientation(mws1gt_4,0,134). coords(mws1gt_4,27,140,134).  visible(mws1gt_4,134). happensAt(walking(mws1gt_4),134).
orientation(mws1gt_1,155,135). coords(mws1gt_1,125,154,135).  visible(mws1gt_1,135). happensAt(walking(mws1gt_1),135). orientation(mws1gt_4,0,135). coords(mws1gt_4,27,140,135).  visible(mws1gt_4,135). happensAt(walking(mws1gt_4),135).
orientation(mws1gt_1,155,136). coords(mws1gt_1,127,154,136).  visible(mws1gt_1,136). happensAt(walking(mws1gt_1),136). orientation(mws1gt_4,0,136). coords(mws1gt_4,27,140,136).  visible(mws1gt_4,136). happensAt(walking(mws1gt_4),136).
orientation(mws1gt_1,155,137). coords(mws1gt_1,128,154,137).  visible(mws1gt_1,137). happensAt(walking(mws1gt_1),137). orientation(mws1gt_4,0,137). coords(mws1gt_4,27,138,137).  visible(mws1gt_4,137). happensAt(walking(mws1gt_4),137).
orientation(mws1gt_1,155,138). coords(mws1gt_1,129,153,138).  visible(mws1gt_1,138). happensAt(walking(mws1gt_1),138). orientation(mws1gt_4,0,138). coords(mws1gt_4,27,138,138).  visible(mws1gt_4,138). happensAt(walking(mws1gt_4),138).
#show holdsAt/2.