
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

	orientation(lb1gt_0,156,36). coords(lb1gt_0,177,129,36).  visible(lb1gt_0,36). happensAt(walking(lb1gt_0),36). orientation(lb1gt_2,142,36). coords(lb1gt_2,80,75,36).  visible(lb1gt_2,36). happensAt(walking(lb1gt_2),36).
orientation(lb1gt_0,150,37). coords(lb1gt_0,178,128,37).  visible(lb1gt_0,37). happensAt(walking(lb1gt_0),37). orientation(lb1gt_2,142,37). coords(lb1gt_2,80,75,37).  visible(lb1gt_2,37). happensAt(walking(lb1gt_2),37).
orientation(lb1gt_0,150,38). coords(lb1gt_0,178,128,38).  visible(lb1gt_0,38). happensAt(walking(lb1gt_0),38). orientation(lb1gt_2,142,38). coords(lb1gt_2,81,75,38).  visible(lb1gt_2,38). happensAt(walking(lb1gt_2),38).
orientation(lb1gt_0,150,39). coords(lb1gt_0,179,127,39).  visible(lb1gt_0,39). happensAt(walking(lb1gt_0),39). orientation(lb1gt_2,142,39). coords(lb1gt_2,81,75,39).  visible(lb1gt_2,39). happensAt(walking(lb1gt_2),39).
orientation(lb1gt_0,150,40). coords(lb1gt_0,180,126,40).  visible(lb1gt_0,40). happensAt(walking(lb1gt_0),40). orientation(lb1gt_2,142,40). coords(lb1gt_2,81,75,40).  visible(lb1gt_2,40). happensAt(walking(lb1gt_2),40).
orientation(lb1gt_0,150,41). coords(lb1gt_0,180,126,41).  visible(lb1gt_0,41). happensAt(walking(lb1gt_0),41). orientation(lb1gt_2,142,41). coords(lb1gt_2,81,76,41).  visible(lb1gt_2,41). happensAt(walking(lb1gt_2),41).
orientation(lb1gt_0,150,42). coords(lb1gt_0,181,124,42).  visible(lb1gt_0,42). happensAt(walking(lb1gt_0),42). orientation(lb1gt_2,142,42). coords(lb1gt_2,81,76,42).  visible(lb1gt_2,42). happensAt(walking(lb1gt_2),42).
orientation(lb1gt_0,150,43). coords(lb1gt_0,182,123,43).  visible(lb1gt_0,43). happensAt(walking(lb1gt_0),43). orientation(lb1gt_2,142,43). coords(lb1gt_2,82,76,43).  visible(lb1gt_2,43). happensAt(walking(lb1gt_2),43).
orientation(lb1gt_0,150,44). coords(lb1gt_0,182,122,44).  visible(lb1gt_0,44). happensAt(walking(lb1gt_0),44). orientation(lb1gt_2,142,44). coords(lb1gt_2,82,76,44).  visible(lb1gt_2,44). happensAt(walking(lb1gt_2),44).
orientation(lb1gt_0,148,45). coords(lb1gt_0,182,121,45).  visible(lb1gt_0,45). happensAt(walking(lb1gt_0),45). orientation(lb1gt_2,142,45). coords(lb1gt_2,82,76,45).  visible(lb1gt_2,45). happensAt(walking(lb1gt_2),45).
orientation(lb1gt_0,148,46). coords(lb1gt_0,182,121,46).  visible(lb1gt_0,46). happensAt(walking(lb1gt_0),46). orientation(lb1gt_2,142,46). coords(lb1gt_2,82,76,46).  visible(lb1gt_2,46). happensAt(walking(lb1gt_2),46).
orientation(lb1gt_0,148,47). coords(lb1gt_0,182,120,47).  visible(lb1gt_0,47). happensAt(walking(lb1gt_0),47). orientation(lb1gt_2,142,47). coords(lb1gt_2,82,76,47).  visible(lb1gt_2,47). happensAt(walking(lb1gt_2),47).
orientation(lb1gt_0,148,48). coords(lb1gt_0,182,119,48).  visible(lb1gt_0,48). happensAt(walking(lb1gt_0),48). orientation(lb1gt_2,142,48). coords(lb1gt_2,82,76,48).  visible(lb1gt_2,48). happensAt(walking(lb1gt_2),48).
orientation(lb1gt_0,148,49). coords(lb1gt_0,182,119,49).  visible(lb1gt_0,49). happensAt(walking(lb1gt_0),49). orientation(lb1gt_2,142,49). coords(lb1gt_2,82,76,49).  visible(lb1gt_2,49). happensAt(walking(lb1gt_2),49).
orientation(lb1gt_0,148,50). coords(lb1gt_0,182,119,50).  visible(lb1gt_0,50). happensAt(walking(lb1gt_0),50). orientation(lb1gt_2,142,50). coords(lb1gt_2,82,76,50).  visible(lb1gt_2,50). happensAt(walking(lb1gt_2),50).
orientation(lb1gt_0,148,51). coords(lb1gt_0,182,119,51).  visible(lb1gt_0,51). happensAt(walking(lb1gt_0),51). orientation(lb1gt_2,142,51). coords(lb1gt_2,82,77,51).  visible(lb1gt_2,51). happensAt(walking(lb1gt_2),51).
orientation(lb1gt_0,148,52). coords(lb1gt_0,182,119,52).  visible(lb1gt_0,52). happensAt(walking(lb1gt_0),52). orientation(lb1gt_2,142,52). coords(lb1gt_2,82,78,52).  visible(lb1gt_2,52). happensAt(walking(lb1gt_2),52).
orientation(lb1gt_0,148,53). coords(lb1gt_0,182,119,53).  visible(lb1gt_0,53). happensAt(walking(lb1gt_0),53). orientation(lb1gt_2,142,53). coords(lb1gt_2,82,78,53).  visible(lb1gt_2,53). happensAt(walking(lb1gt_2),53).
orientation(lb1gt_0,148,54). coords(lb1gt_0,182,119,54).  visible(lb1gt_0,54). happensAt(walking(lb1gt_0),54). orientation(lb1gt_2,142,54). coords(lb1gt_2,82,78,54).  visible(lb1gt_2,54). happensAt(walking(lb1gt_2),54).
orientation(lb1gt_0,148,55). coords(lb1gt_0,182,119,55).  visible(lb1gt_0,55). happensAt(walking(lb1gt_0),55). orientation(lb1gt_2,142,55). coords(lb1gt_2,82,78,55).  visible(lb1gt_2,55). happensAt(walking(lb1gt_2),55).
orientation(lb1gt_0,148,56). coords(lb1gt_0,181,119,56).  visible(lb1gt_0,56). happensAt(walking(lb1gt_0),56). orientation(lb1gt_2,142,56). coords(lb1gt_2,82,79,56).  visible(lb1gt_2,56). happensAt(walking(lb1gt_2),56).
orientation(lb1gt_0,148,57). coords(lb1gt_0,181,119,57).  visible(lb1gt_0,57). happensAt(active(lb1gt_0),57). orientation(lb1gt_2,142,57). coords(lb1gt_2,83,79,57).  visible(lb1gt_2,57). happensAt(walking(lb1gt_2),57).
orientation(lb1gt_0,148,58). coords(lb1gt_0,181,119,58).  visible(lb1gt_0,58). happensAt(active(lb1gt_0),58). orientation(lb1gt_2,142,58). coords(lb1gt_2,83,79,58).  visible(lb1gt_2,58). happensAt(walking(lb1gt_2),58).
orientation(lb1gt_0,148,59). coords(lb1gt_0,181,119,59).  visible(lb1gt_0,59). happensAt(active(lb1gt_0),59). orientation(lb1gt_2,142,59). coords(lb1gt_2,83,79,59).  visible(lb1gt_2,59). happensAt(walking(lb1gt_2),59).
orientation(lb1gt_0,148,60). coords(lb1gt_0,181,119,60).  visible(lb1gt_0,60). happensAt(active(lb1gt_0),60). orientation(lb1gt_2,142,60). coords(lb1gt_2,83,79,60).  visible(lb1gt_2,60). happensAt(walking(lb1gt_2),60).
#show holdsAt/2.