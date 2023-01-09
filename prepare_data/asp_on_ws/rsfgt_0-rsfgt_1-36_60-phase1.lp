
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

	orientation(rsfgt_0,160,36). coords(rsfgt_0,103,127,36).  visible(rsfgt_0,36). happensAt(walking(rsfgt_0),36). orientation(rsfgt_1,150,36). coords(rsfgt_1,185,152,36).  visible(rsfgt_1,36). happensAt(walking(rsfgt_1),36).
orientation(rsfgt_0,160,37). coords(rsfgt_0,102,127,37).  visible(rsfgt_0,37). happensAt(walking(rsfgt_0),37). orientation(rsfgt_1,150,37). coords(rsfgt_1,185,149,37).  visible(rsfgt_1,37). happensAt(walking(rsfgt_1),37).
orientation(rsfgt_0,160,38). coords(rsfgt_0,102,127,38).  visible(rsfgt_0,38). happensAt(walking(rsfgt_0),38). orientation(rsfgt_1,150,38). coords(rsfgt_1,185,146,38).  visible(rsfgt_1,38). happensAt(walking(rsfgt_1),38).
orientation(rsfgt_0,160,39). coords(rsfgt_0,99,127,39).  visible(rsfgt_0,39). happensAt(walking(rsfgt_0),39). orientation(rsfgt_1,150,39). coords(rsfgt_1,185,144,39).  visible(rsfgt_1,39). happensAt(walking(rsfgt_1),39).
orientation(rsfgt_0,160,40). coords(rsfgt_0,96,128,40).  visible(rsfgt_0,40). happensAt(walking(rsfgt_0),40). orientation(rsfgt_1,150,40). coords(rsfgt_1,185,141,40).  visible(rsfgt_1,40). happensAt(walking(rsfgt_1),40).
orientation(rsfgt_0,160,41). coords(rsfgt_0,94,129,41).  visible(rsfgt_0,41). happensAt(walking(rsfgt_0),41). orientation(rsfgt_1,150,41). coords(rsfgt_1,185,141,41).  visible(rsfgt_1,41). happensAt(walking(rsfgt_1),41).
orientation(rsfgt_0,160,42). coords(rsfgt_0,93,131,42).  visible(rsfgt_0,42). happensAt(walking(rsfgt_0),42). orientation(rsfgt_1,150,42). coords(rsfgt_1,185,139,42).  visible(rsfgt_1,42). happensAt(walking(rsfgt_1),42).
orientation(rsfgt_0,160,43). coords(rsfgt_0,93,131,43).  visible(rsfgt_0,43). happensAt(walking(rsfgt_0),43). orientation(rsfgt_1,150,43). coords(rsfgt_1,184,138,43).  visible(rsfgt_1,43). happensAt(walking(rsfgt_1),43).
orientation(rsfgt_0,160,44). coords(rsfgt_0,92,131,44).  visible(rsfgt_0,44). happensAt(walking(rsfgt_0),44). orientation(rsfgt_1,150,44). coords(rsfgt_1,184,137,44).  visible(rsfgt_1,44). happensAt(walking(rsfgt_1),44).
orientation(rsfgt_0,160,45). coords(rsfgt_0,91,131,45).  visible(rsfgt_0,45). happensAt(walking(rsfgt_0),45). orientation(rsfgt_1,150,45). coords(rsfgt_1,184,137,45).  visible(rsfgt_1,45). happensAt(walking(rsfgt_1),45).
orientation(rsfgt_0,160,46). coords(rsfgt_0,92,132,46).  visible(rsfgt_0,46). happensAt(walking(rsfgt_0),46). orientation(rsfgt_1,150,46). coords(rsfgt_1,184,136,46).  visible(rsfgt_1,46). happensAt(walking(rsfgt_1),46).
orientation(rsfgt_0,160,47). coords(rsfgt_0,91,132,47).  visible(rsfgt_0,47). happensAt(walking(rsfgt_0),47). orientation(rsfgt_1,150,47). coords(rsfgt_1,184,136,47).  visible(rsfgt_1,47). happensAt(walking(rsfgt_1),47).
orientation(rsfgt_0,160,48). coords(rsfgt_0,90,134,48).  visible(rsfgt_0,48). happensAt(walking(rsfgt_0),48). orientation(rsfgt_1,150,48). coords(rsfgt_1,183,134,48).  visible(rsfgt_1,48). happensAt(walking(rsfgt_1),48).
orientation(rsfgt_0,160,49). coords(rsfgt_0,90,134,49).  visible(rsfgt_0,49). happensAt(walking(rsfgt_0),49). orientation(rsfgt_1,150,49). coords(rsfgt_1,183,133,49).  visible(rsfgt_1,49). happensAt(walking(rsfgt_1),49).
orientation(rsfgt_0,160,50). coords(rsfgt_0,89,134,50).  visible(rsfgt_0,50). happensAt(walking(rsfgt_0),50). orientation(rsfgt_1,150,50). coords(rsfgt_1,184,133,50).  visible(rsfgt_1,50). happensAt(walking(rsfgt_1),50).
orientation(rsfgt_0,160,51). coords(rsfgt_0,87,134,51).  visible(rsfgt_0,51). happensAt(walking(rsfgt_0),51). orientation(rsfgt_1,150,51). coords(rsfgt_1,184,131,51).  visible(rsfgt_1,51). happensAt(walking(rsfgt_1),51).
orientation(rsfgt_0,160,52). coords(rsfgt_0,85,134,52).  visible(rsfgt_0,52). happensAt(walking(rsfgt_0),52). orientation(rsfgt_1,150,52). coords(rsfgt_1,183,128,52).  visible(rsfgt_1,52). happensAt(walking(rsfgt_1),52).
orientation(rsfgt_0,160,53). coords(rsfgt_0,82,134,53).  visible(rsfgt_0,53). happensAt(walking(rsfgt_0),53). orientation(rsfgt_1,150,53). coords(rsfgt_1,183,127,53).  visible(rsfgt_1,53). happensAt(walking(rsfgt_1),53).
orientation(rsfgt_0,160,54). coords(rsfgt_0,82,134,54).  visible(rsfgt_0,54). happensAt(walking(rsfgt_0),54). orientation(rsfgt_1,150,54). coords(rsfgt_1,182,127,54).  visible(rsfgt_1,54). happensAt(walking(rsfgt_1),54).
orientation(rsfgt_0,160,55). coords(rsfgt_0,81,134,55).  visible(rsfgt_0,55). happensAt(walking(rsfgt_0),55). orientation(rsfgt_1,150,55). coords(rsfgt_1,179,126,55).  visible(rsfgt_1,55). happensAt(walking(rsfgt_1),55).
orientation(rsfgt_0,160,56). coords(rsfgt_0,80,134,56).  visible(rsfgt_0,56). happensAt(walking(rsfgt_0),56). orientation(rsfgt_1,150,56). coords(rsfgt_1,178,126,56).  visible(rsfgt_1,56). happensAt(walking(rsfgt_1),56).
orientation(rsfgt_0,160,57). coords(rsfgt_0,80,134,57).  visible(rsfgt_0,57). happensAt(walking(rsfgt_0),57). orientation(rsfgt_1,150,57). coords(rsfgt_1,178,123,57).  visible(rsfgt_1,57). happensAt(walking(rsfgt_1),57).
orientation(rsfgt_0,160,58). coords(rsfgt_0,80,134,58).  visible(rsfgt_0,58). happensAt(walking(rsfgt_0),58). orientation(rsfgt_1,150,58). coords(rsfgt_1,178,122,58).  visible(rsfgt_1,58). happensAt(walking(rsfgt_1),58).
orientation(rsfgt_0,160,59). coords(rsfgt_0,80,136,59).  visible(rsfgt_0,59). happensAt(walking(rsfgt_0),59). orientation(rsfgt_1,150,59). coords(rsfgt_1,177,121,59).  visible(rsfgt_1,59). happensAt(walking(rsfgt_1),59).
orientation(rsfgt_0,160,60). coords(rsfgt_0,79,136,60).  visible(rsfgt_0,60). happensAt(walking(rsfgt_0),60). orientation(rsfgt_1,150,60). coords(rsfgt_1,177,117,60).  visible(rsfgt_1,60). happensAt(walking(rsfgt_1),60).
#show holdsAt/2.